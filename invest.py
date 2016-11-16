#!/usr/bin/python
# TODO:
#  - price and dividend history
#  - plots
#  - handle splits
#  - cash
#  - dividend reinvestment (need fractional shares?)
#  - fees

import argparse
import csv
import datetime
import itertools
import math
import numpy
import os
import os.path
import sqlite3
import string
import sys
import urllib2

class NetworkError(Exception):
  pass

def _annualize(r, t):
  return math.pow(1+r, 360.0 / t.days) - 1

def _str2date(s):
  return datetime.date(*[int(x) for x in s.split('-')])

def _rows(c):
  r = c.fetchone()
  while r:
    yield r
    r = c.fetchone()

def _dbpath(name):
  name += '.db'
  base = os.path.join(os.path.expanduser('~'), ".invest")
  if not os.path.isdir(base):
    os.mkdir(base)
  return os.path.join(base, name)

class Cache(object):
  def __init__(self):
    self.db = sqlite3.connect(_dbpath('_cache'), detect_types=sqlite3.PARSE_DECLTYPES|sqlite3.PARSE_COLNAMES)
    self.db.execute("create table if not exists prices (date date, sym text, open real, close real)")
    self.db.execute("create table if not exists dividends (date date, sym text, amt real)")
    self.db.execute("create table if not exists splits (date date, sym text, ratio real)")
    self.db.commit()
    self.updated = dict()
    self.can_download = True

  def get_open(self, sym, date):
    return self._get("open", sym, date)
  def get_close(self, sym, date):
    return self._get("close", sym, date)
  def get_price(self, sym, date):
    return self.get_close(sym, date)

  def _get(self, what, sym, date):
    self._update(sym, date)
    c = self.db.cursor()
    c.execute('select %s, date as "date[date]" from prices where upper(sym) = ? and date <= ? order by date desc limit 1' % (what,), (sym.upper(), date))
    return c.fetchone()

  def get_dividends(self, syms, start, end):
    c = self.db.cursor()
    parms = [start, end] + [s.upper() for s in syms]
    qs = ",".join('?' * len(syms))
    c.execute('select date, upper(sym), amt from dividends where date between ? and ? and upper(sym) in (%s) order by date' % qs, parms)
    return _rows(c)

  def _update(self, sym, date):
    c = self.db.cursor()
    c.execute('select max(date) as "date[date]" from prices where upper(sym) = ?', (sym.upper(),))
    most_recent = c.fetchone()[0]
    if most_recent:
      most_recent += datetime.timedelta(days=1)
    if not most_recent or most_recent <= date:
      self._download(sym, most_recent)

  def _download(self, sym, start=None):
    if not self.can_download or sym.upper() in self.updated:
      return
    self.updated[sym.upper()] = True
    print "download", sym
    base = 'http://ichart.finance.yahoo.com/{0}?s={1}'
    if start != None:
      base += ('&a={0}&b={1}&c={2}').format(start.month-1, start.day, start.year)
    url = base.format('table.csv', sym)
    print url
    try:
      price_src = urllib2.urlopen(url)
    except urllib2.HTTPError, e:
      if e.code == 404:
        print "Failed to retrieve data for %s" % sym
        return
      else:
        raise
    price_src.readline()
    reader = csv.reader(price_src)
    for r in reader:
      self.db.execute('insert into prices values (?,?,?,?)', (r[0], sym, r[1], r[4]))
    self.db.commit()

    url = base.format('x', sym) + '&g=v'
    try:
      div_src = urllib2.urlopen(url)
    except urllib2.HTTPError,e:
      if e.code == 404:
        print "Failed to retrieve data for %s" % sym
        return
      else:
        raise
    div_src.readline()
    reader = csv.reader(div_src)
    for r in reader:
      if r[0] == 'DIVIDEND':
        date = datetime.datetime.strptime(string.strip(r[1]), '%Y%m%d').date()
        self.db.execute('insert into dividends values (?,?,?)', (date, sym, r[2]))
      elif r[0] == 'SPLIT':
        date = datetime.datetime.strptime(string.strip(r[1]), '%Y%m%d').date()
        split = string.split(r[2], ':', 1)
        ratio = float(split[1]) / float(split[0])
        self.db.execute('insert into splits values (?,?,?)', (date, sym, ratio))
    self.db.commit()

cache = Cache()

def _get_portfolio_db(name):
  path = _dbpath(name)
  if not os.path.isfile(path):
    return None
  return _init_portfolio_db(name)

def _init_portfolio_db(name):
  db = sqlite3.connect(_dbpath(name), detect_types=sqlite3.PARSE_DECLTYPES)
  db.execute("create table if not exists transactions (date date, sym text, num int, price real)")
  db.execute("create table if not exists tags (name text primary key, parent text, weight real)")
  db.execute("create table if not exists weights (sym text, tag text, weight real, primary key (sym, tag))")
  db.execute("create table if not exists preferences (tag text primary key, sym text)")
  db.execute("insert or ignore into tags values (?, '', 1)", (name,))
  db.commit()
  return db

def _get_tags(db, parent = None):
  c = db.cursor()
  if parent == None:
    c.execute('select name, parent, weight from tags order by parent')
  else:
    c.execute('select name, parent, weight from tags where upper(parent) = ?', (parent.upper(),))
  tags = dict()
  for r in _rows(c):
    name, parent, weight = r
    if parent not in tags:
      tags[parent] = dict()
    tags[parent][name] = weight
  return tags

def _get_asset_tags(db, sym):
  c = db.cursor()
  c.execute('select tag, weight from weights where upper(sym) = ? order by tag', (sym.upper(),))
  return { r[0]: r[1] for r in _rows(c) }

def _validate_tag(db, tag):
  c = db.cursor()
  c.execute('select count(*) from tags where name = ?', (tag,))
  if c.fetchone()[0] == 0:
    raise ValueError('Unknown tag %s' % tag)

def _get_preferences(db):
  c = db.cursor()
  c.execute('select tag, sym from preferences order by tag')
  return { r[0]: r[1] for r in _rows(c) }

class Tool(object):
  def usage(self, cmd): self._get_parser(cmd).print_help()
  def _get_parser(self, cmd):
    parser = self._make_parser(cmd);
    self._add_args(parser)
    return parser
  def _add_date_arg(self, parser, name, default, help_text):
    parser.add_argument(name, type=_str2date, default=default, metavar='YYYY-MM-DD', help=help_text)
  def _add_date_args(self, parser):
    self._add_date_arg(parser, '--start', datetime.date(1,1,1), "start date")
    self._add_date_arg(parser, '--end', datetime.date.today(), "end date")
  def _make_parser(self, cmd):
    return argparse.ArgumentParser(prog=("{0} {1}").format(cmd, self.name()),
                                   description=self.description(),
                                   add_help=False)

class NewPortfolioTool(Tool):
  def name(self): return "new-portfolio"
  def description(self): return "initialize a new investment portfolio"
  def _add_args(self, parser):
    parser.add_argument('name', type=str, help="name of the portfolio")

  def run(self, cmd, args):
    opts = self._get_parser(cmd).parse_args(args)
    if _get_portfolio_db(opts.name) != None:
      raise ValueError("A portfolio with that name already exists!")
    else:
      _init_portfolio_db(opts.name)

class PortfolioTool(Tool):
  def _add_common_args(self, parser):
    parser.add_argument('portfolio', type=str, help="portfolio of interest")
  def _add_download_args(self, parser):
    parser.add_argument('--download', dest='download', action='store_true', help="enable downloading data")
    parser.add_argument('--no-download', dest='download', action='store_false', help="enable downloading data")
    parser.set_defaults(download=True)
  def run(self, cmd, args):
    opts = self._get_parser(cmd).parse_args(args)
    if 'download' in opts:
      cache.can_download = bool(opts.download)
    db = _get_portfolio_db(opts.portfolio)
    self._run(db, opts)

class ValueTool(Tool):
  def name(self): return "value"
  def description(self): return "list equity value"
  def _add_args(self, parser):
    self._add_download_args(parser)
    parser.add_argument('symbol', type=str, help="ticker symbol")
    self._add_date_arg(parser, '--date', datetime.date.today(), "date of interest")
  def run(self, cmd, args):
    opts = self._get_parser(cmd).parse_args(args)
    price = cache.get_price(opts.symbol, opts.date)
    print ("{1}: ${0:.2f}").format(*price)

class TransactionTool(PortfolioTool):
  def __init__(self, verb, past_verb, noun):
    self.verb, self.past_verb, self.noun = verb, past_verb, noun
  def name(self): return self.verb
  def description(self): return "record an equity %s" % self.noun
  def usage(self, cmd): self._get_parser(cmd).print_help()
  def _add_args(self, parser):
    self._add_common_args(parser)
    parser.add_argument('symbol', type=str, help="the equity's symbol")
    parser.add_argument('quantity', type=int, help="number of shares")
    parser.add_argument('price', type=float, help="per-share price")
    self._add_date_arg(parser, '--date', datetime.date.today(), "transaction date")
    return parser

class BuyTool(TransactionTool):
  def __init__(self):
    super(BuyTool, self).__init__("buy", "bought", "purchase")
  def _run(self, db, opts):
    db.execute("insert into transactions values (?,?,?,?)", (opts.date, opts.symbol.upper(), opts.quantity, opts.price))
    db.commit()

class SellTool(TransactionTool):
  def __init__(self):
    super(SellTool, self).__init__("sell", "sold", "sale")
  def _run(self, db, opts):
    db.execute("insert into transactions values (?,?,?,?)", (opts.date, opts.symbol.upper(), -opts.quantity, opts.price))
    db.commit()

class CancelTool(PortfolioTool):
  def name(self): return "cancel"
  def description(self): return "cancel a transaction"
  def _add_args(self, parser):
    self._add_common_args(parser)
    parser.add_argument('id', type=int, help="transaction id")
  def _run(self, db, opts):
    db.execute('delete from transactions where rowid=?', (opts.id,))
    db.commit()

class LedgerTool(PortfolioTool):
  def name(self): return "ledger"
  def description(self): return "list portfolio transactions"
  def _add_args(self, parser):
    self._add_common_args(parser)
    self._add_date_args(parser)
  def _run(self, db, opts):
    c = db.cursor()
    c.execute('select rowid, date, sym, num, price from transactions where date between ? and ? order by date', (opts.start, opts.end))
    for r in _rows(c):
      print ("({0})\t{1}: {2: >8}{3: >-6,} @ ${4: >-7,.3f}").format(*r)

class TagTool(PortfolioTool):
  def name(self): return "tags"
  def description(self): return "Adjust asset class tags"
  def _add_args(self, parser):
    self._add_common_args(parser)
    parser.add_argument("parent", type=str, nargs='?', help="parent category (or root)")
    parser.add_argument("tags", type=str, nargs='*', help="tag, weight pairs")
  def _run(self, db, opts):
    if len(opts.tags) == 0:
      tags = _get_tags(db, opts.parent)
      if opts.parent == None:
        opts.parent = opts.portfolio
      self._show(tags, opts.parent, 1, 1)
    else:
      self._update(db, opts.parent, opts.tags)

  def _show(self, tags, cur, running, depth):
    if cur not in tags:
      return
    children = tags[cur]
    print "%s:" % cur
    for child, weight in children.iteritems():
      print ("{3}{0} = {1:.2f}% ({2:.2f}% cumulative)").format(child, 100 * weight, 100 * weight * running, ' ' * (2*depth))
      self._show(tags, child, weight * running, depth + 1)

  def _update(self, db, parent, tags):
    it = iter(tags)
    for tag in it:
      tag = "%s/%s" % (parent, tag)
      _validate_tag(db, tag)
      weight = next(it)
      db.execute('insert or replace into tags values (?,?,?)', (tag, parent, weight))
    db.commit()
    tags = _get_tags(db, parent)[parent]
    tot = sum([ w for _,w in tags.iteritems() ])
    if tot != 1:
      print "Allocations do not sum to 100%. Rebalancing to:"
      for tag, weight in tags.iteritems():
        weight /= tot
        print ("  {0} = {1:.2f}%").format(tag, 100 * weight)
        db.execute('update tags set weight = ? where upper(name) = ? and upper(parent) = ?', (weight, tag.upper(), parent.upper()))
        db.commit()

class WeightTool(PortfolioTool):
  def name(self): return "weights"
  def description(self): return "Set tag weights for a symbol"
  def _add_args(self, parser):
    self._add_common_args(parser)
    parser.add_argument("symbol", type=str, help="symbol of interest")
    parser.add_argument("weights", type=str, nargs='*', help="tag, weight pairs")
  def _run(self, db, opts):
    if len(opts.weights) == 0:
      self._show_weights(db, opts.symbol)
    else:
      self._update_weights(db, opts.portfolio, opts.symbol, opts.weights)

  def _show_weights(self, db, sym):
    weights = _get_asset_tags(db, sym)
    for tag, weight in weights.iteritems():
      print ("  {0} = {1:.2f}%").format(tag, 100 * weight)

  def _update_weights(self, db, portfolio, sym, args):
    it = iter(args)
    tot = 0
    weights = dict()
    for tag in it:
      tag = "%s/%s" % (portfolio, tag)
      weight = next(it)
      weights[tag] = weight
      tot += float(weight)
      db.execute('insert or replace into weights values (?,?,?)', (sym, tag, weight))
    db.commit()

    if tot != 1:
      print "Weights do not sum to 100%. Rebalancing to:"
      for tag, w in weights.iteritems():
        w = float(w) / tot
        print ("  {0} = {1:.2f}%").format(tag, 100 * w)
        db.execute('update weights set weight = ? where upper(sym) = ? and upper(tag) = ?', (w, sym.upper(), tag.upper()))
        db.commit()

class BalanceTool(PortfolioTool):
  def name(self): return "balance"
  def description(self): return "show tag balance and propose rebalancing"
  def _add_args(self, parser):
    self._add_common_args(parser)
    self._add_download_args(parser)
    parser.add_argument('--invest', type=float, default=0, help="incorporate addition or removal of funds")
  def _run(self, db, opts):
    date = datetime.date.today()
    c = db.cursor()
    c.execute('select sym, sum(num) from transactions group by sym')
    tot_value = opts.invest
    balance = dict()
    for r in _rows(c):
      sym, num = r
      tags = _get_asset_tags(db, sym)
      price = cache.get_price(sym, date)[0]
      value = num * price
      tot_value += value
      for tag, weight in tags.iteritems():
        if tag not in balance:
          balance[tag] = 0
        balance[tag] += weight * value
    print ("Current valuation: ${0:.2f}").format(tot_value)
    print "Current balance:"
    for tag, value in balance.iteritems():
      print ("    {0}: ${1:.2f} ({2:.2f}%)").format(tag, value, 100 * value / tot_value)

    corbalance = dict()
    calibration = dict()
    tags = _get_tags(db)
    corbalance = self._get_weights(tags, opts.portfolio, 1, corbalance)
    print "Desired balance:"
    for tag, weight in corbalance.iteritems():
      print ("    {0}: ${1:.2f} ({2:.2f}%)").format(tag, weight * tot_value, 100 * weight)
      if tag in balance:
        calibration[tag] = weight * tot_value - balance[tag]
      else:
        calibration[tag] = weight * tot_value
    for tag, value in balance.iteritems():
      if tag not in corbalance:
        calibration[tag] = -value

    print "Required re-calibration:"
    for tag, calib in calibration.iteritems():
      print ("  {0}: ${1:.2f}").format(tag, calib)

    c.execute('select sym, tag, weight from weights')
    prices = dict()
    weights = dict()
    for r in _rows(c):
      sym, tag, weight = r
      print sym, tag, weight
      if tag in calibration:
        if sym not in weights:
          weights[sym] = dict()
        if sym not in prices:
          prices[sym] = cache.get_price(sym, date)[0]
        weights[sym][tag] = weight

    transactions = dict()
    prefs = _get_preferences(db)
    for tag, calib in calibration.iteritems():
      if tag in prefs:
        sym = prefs[tag]
        if sym not in transactions:
          transactions[sym] = 0
        transactions[sym] += calib / prices[sym]
        calibration[tag] = 0

    print "Preference-based transactions:"
    for sym, num in transactions.iteritems():
      print ("  {0}: {1:.2f} @ ${2:.2f} = ${3:.2f}").format(sym, num, prices[sym], num * prices[sym])

    W = numpy.zeros((len(calibration), len(weights)))
    c = numpy.zeros(len(calibration))
    i = 0
    for tag in calibration.iterkeys():
      c[i] = calibration[tag]
      j = 0
      for sym, ws in weights.iteritems():
        if tag in ws:
          W[i,j] = prices[sym] * weights[sym][tag]
        j += 1
      i += 1
    W = numpy.linalg.pinv(W)
    x = W.dot(c)

    print "Suggested transactions:"
    i = 0
    for sym in weights.iterkeys():
      if x[i] != 0:
        print ("  {0}: {1:.2f} @ ${2:.2f} = ${3:.2f}").format(sym, x[i], prices[sym], x[i] * prices[sym])
      i += 1

  def _get_weights(self, tags, cur, curweight, weights):
    if cur not in tags:
      weights[cur] = curweight
      return weights
    children = tags[cur]
    for child, weight in children.iteritems():
      w = weight * curweight
      if w > 0:
        weights = self._get_weights(tags, child, w, weights)
    return weights

class PreferenceTool(PortfolioTool):
  def name(self): return "preferences"
  def description(self): return "manage preferred investments for tags"
  def _add_args(self, parser):
    self._add_common_args(parser)
    parser.add_argument("prefs", type=str, nargs='*', help="tag, symbol pairs")
  def _run(self, db, opts):
    if len(opts.prefs) ==0:
      self._show(db)
    else:
      self._update(db, opts.portfolio, opts.prefs)
  def _show(self, db):
    prefs = _get_preferences(db)
    for tag, sym in prefs.iteritems():
      print "%s: %s" % (tag, sym)
  def _update(self, db, portfolio, prefs):
    it = iter(prefs)
    for tag in it:
      tag = "%s/%s" % (portfolio, tag)
      _validate_tag(db, tag)
      sym = next(it)
      db.execute('insert or replace into preferences values (?,?)', (tag, sym))
    db.commit()

class ImportTool(PortfolioTool):
  def name(self): return "import"
  def description(self): return "import CSV file of transactions"
  def _make_parser(self, cmd):
    return argparse.ArgumentParser(prog=("{0} {1}").format(cmd, self.name()),
                                   description=self.description()+".\nOne transaction per row: date,symbol,quantity,price-per",
                                   formatter_class=argparse.RawTextHelpFormatter,
                                   add_help=False)
  def _add_args(self, parser):
    self._add_common_args(parser)
    parser.add_argument('file', type=str, help="file to import")
  def _run(self, db, opts):
    f = open(opts.file, 'rb')
    reader = csv.reader(f)
    for r in reader:
      db.execute('insert into transactions values (?,?,?,?)', tuple(r))
    db.commit()

class ExportTool(PortfolioTool):
  def name(self): return "export"
  def description(self): return "export transactions to CSV file"
  def _add_args(self, parser):
    self._add_common_args(parser)
    parser.add_argument('file', type=str, help="file to import")
    self._add_date_args(parser)
  def _run(self, db, opts):
    f = open(opts.file, 'wb')
    writer = csv.writer(f)
    c = db.cursor()
    c.execute('select date, sym, num, price from transactions where date between ? and ? order by date', (opts.start, opts.end))
    writer.writerows(_rows(c))

class PositionTool(PortfolioTool):
  def name(self): return "position"
  def description(self): return "summarize accumulated position"
  def _add_args(self, parser):
    self._add_common_args(parser)
    self._add_download_args(parser)
    self._add_date_arg(parser, '--end', datetime.date.today(), "end date")
  def _run(self, db, opts):
    c = db.cursor()
    price = dict()
    num = dict()
    realized = dict()
    c.execute('select sym, num, price from transactions where date <= ? order by date', (opts.end,))
    for r in _rows(c):
      s, n, p = r
      if s not in num: num[s] = 0
      if s not in price: price[s] = 0
      if s not in realized: realized[s] = 0
      if n < 0:
        loss = n * (p - (price[s]/num[s]))
        realized[s] -= loss
        price[s] += n * price[s]
      else:
        price[s] += n * p
      num[s] += n

    total_realized = 0
    total_unrealized = 0
    for sym in price.iterkeys():
      print ("{0: <7} {1: >-6,} @ ${2: >-7,.3f}").format(sym, num[sym], price[sym] / num[sym])
      cur_price = cache.get_price(sym, opts.end)[0]
      cur_value = cur_price * num[sym]
      print ("         Current value {0: >-6,} @ ${1:.2f} = ${2:.2f}").format(num[sym], cur_price, cur_value)
      unrealized = cur_value - price[sym]
      print ("         Realized: ${0:.2f}").format(realized[sym])
      print ("         Unrealized: ${0:.2f}").format(unrealized)
      total_realized += realized[s]
      total_unrealized += unrealized
    print ("Total realized: ${0:.2f}").format(total_realized)
    print ("Total unrealized: ${0:.2f}").format(total_unrealized)

class ReturnTool(PortfolioTool):
  def name(self): return "return"
  def description(self): return "calculate return on investment"
  def _add_args(self, parser):
    self._add_common_args(parser)
    self._add_download_args(parser)
    self._add_date_args(parser)
    parser.add_argument('--symbol', type=str, help="symbol of interest")
  def _run(self, db, opts):
    if opts.symbol:
      where = "= '%s'" % opts.symbol.upper()
    else:
      where = 'not null'

    roi = 1
    growth = 1
    tot_yield = 1
    tot_div = 0
    date = opts.start

    c = db.cursor()
    c.execute('select upper(sym), sum(num) from transactions where upper(sym) %s and date < ? group by sym' % where, (opts.start,))
    position = { r[0] : r[1] for r in _rows(c) }
    prices = { sym: cache.get_open(sym, date)[0] for sym in position.iterkeys() }
    value = sum([ position[sym] * prices[sym] for sym in position.iterkeys() ])
    initial_value = value
    c.execute('select date, upper(sym), num, price from transactions where upper(sym) %s and date between ? and ?' % where, (opts.start, opts.end))
    for r in _rows(c):
      new_date, sym, num, price = r
      if value == 0:
        opts.start = new_date
        print "shifting start date to %s" % new_date

      if new_date == date:
        value += price * num
      else:
        syms = position.keys()
        prices = { s: cache.get_open(s, new_date)[0] for s in position.iterkeys() }
        new_value = sum([ position[s] * prices[s] for s in position.iterkeys() ])

        prev_div = tot_div
        if len(syms) > 0:
          divs = cache.get_dividends(syms, date, new_date - datetime.timedelta(days=1))
          for div in divs:
            ddate, dsym, dprice = div[0], div[1], div[2]
            div_value = dprice * position[dsym]
            #print (" On {0}, dividend of ${1: >-5.2f} for {2} x {3} = ${4: >-8.2f}").format(ddate, dprice, dsym, position[dsym], div_value)
            new_value += div_value
            tot_div += div_value

        date = new_date
        if value > 0:
          cur_roi = new_value / value
          roi *= cur_roi
          cur_growth = (new_value - tot_div) / (value - prev_div)
          growth *= cur_growth
#          print ("On {0}, ${1: >-10.2f}, component RoI is {2}, running RoI is {3}").format(date, value, cur_roi, roi)
#          print ("Growth {0}, ${1: >-10.2f}, component RoI is {2}, running RoI is {3}").format(new_date, value - tot_div, cur_growth, growth)
        value = new_value + price * num
      if sym not in position:
        position[sym] = 0
      position[sym] += num

    if len(position) == 0:
      print "No activity in this period."
      return

    new_date = opts.end
    syms = position.keys()
    prices = { s: cache.get_close(s, new_date)[0] for s in position.iterkeys() }
    new_value = sum([ position[s] * prices[s] for s in position.iterkeys() ])

    prev_div = tot_div
    if syms and len(syms) > 0:
      divs = cache.get_dividends(syms, date, new_date - datetime.timedelta(days=1))
      for div in divs:
        ddate, dsym, dprice = div[0], div[1], div[2]
        div_value = dprice * position[dsym]
        #print (" On {0}, dividend of ${1: >-5.2f} for {2} x {3} = ${4: >-8.2f}").format(ddate, dprice, dsym, position[dsym], div_value)
        new_value += div_value
        tot_div += div_value

    cur_roi = new_value / value
    roi *= cur_roi
    cur_growth = (new_value - tot_div) / (value - prev_div)
    growth *= cur_growth
    dt = opts.end - opts.start
#    print ("On {0}, ${1: >-10.2f}, component RoI is {2}, running RoI is {3}").format(new_date, value, cur_roi, roi)
#    print ("Growth {0}, ${1: >-10.2f}, component RoI is {2}, running RoI is {3}").format(new_date, value - tot_div, cur_growth, growth)
    roi -= 1
    growth -= 1
    print ("Initial valuation on {0}: ${1:2f}").format(opts.start, initial_value)
    print ("Final valuation on {0}: ${1:2f}").format(opts.end, new_value)
    print ("Total dividends: ${0:2f}").format(tot_div)
    print ("Total RoI: {0:.2f}% ({1:.2f}% annualized)").format(100 * roi, 100 * _annualize(roi, dt))
    print ("Growth component: {0:.2f}% ({1:.2f}% annualized)").format(100 * growth, 100 * _annualize(growth, dt))
    y = roi - growth
    print ("Yield component: {0:.2f}% ({1:.2f}% annualized)").format(100 * y, 100 * _annualize(y, dt))
    y = tot_div / (value - tot_div)
    print ("Trailing yield: {0:.2f}% ({1:.2f}% annualized)").format(100 * y, 100 * _annualize(y, dt))

class HelpTool(object):
  def name(self): return "help"
  def description(self): return "print usage information"
  def usage(self, cmd): print ("{0} {1} tool-name").format(cmd, self.name())
  def run(self, cmd, args):
    tool = None
    if len(args) == 1:
      tool = _get_tool(args[0])
    if tool:
      tool.usage(cmd)
    else:
      print ("usage: {0} tool-name [options...]").format(cmd)
      print "\nInvesting toolkit.\n",
      print "Available tools:"
      width = max([len(tool.name()) for tool in _TOOLS])
      for tool in _TOOLS:
        print ("  {0}{1}{2}").format(tool.name(), ' ' * (1 + width - len(tool.name())), tool.description())

_TOOLS = [
  NewPortfolioTool(),
  LedgerTool(),
  ValueTool(),
  PositionTool(),
  ReturnTool(),
  BalanceTool(),
  BuyTool(),
  SellTool(),
  CancelTool(),
  WeightTool(),
  PreferenceTool(),
  TagTool(),
  ImportTool(),
  ExportTool(),
  HelpTool()
]
def _get_tool(name):
  for tool in _TOOLS:
    if tool.name() == name:
      return tool
  return None

def main():
  cmd = sys.argv[0]
  sys.argv.pop(0)
  tool = None
  if len(sys.argv) > 0:
    tool = _get_tool(sys.argv[0])
    sys.argv.pop(0)
  if not tool:
    tool = _get_tool('help')
  tool.run(cmd, sys.argv)

if __name__ == "__main__":
  main()
