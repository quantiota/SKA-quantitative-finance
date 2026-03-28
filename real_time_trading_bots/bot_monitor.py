"""
SKA Trading Bot Monitor

Watches RESULTS_DIR for CSV result files.
Every new CSV, computes cumulative P&L and sends an email report.

Usage:
    python bot_monitor.py

Setup:
    Set EMAIL_FROM, EMAIL_TO, GMAIL_APP_PASSWORD below.
    Generate a Gmail App Password at:
    https://myaccount.google.com/apppasswords
"""

import glob
import os
import smtplib
import time
from datetime import datetime
from email.mime.text import MIMEText

import pandas as pd

VERSION       = int(os.environ.get('BOT_VERSION', '2'))

RESULTS_DIR   = f"/home/coder/project/Real_Time_SKA_trading/bot_results_v{VERSION}"
CSV_PATTERN   = f"bot_results_v{VERSION}_*.csv"
POLL_INTERVAL = 300   # seconds
SMTP_SERVER   = "smtp.gmail.com"
SMTP_PORT     = 587

EMAIL_FROM         = "bouarfa.mahi@gmail.com"
EMAIL_TO           = "mairifain@gmail.com"
GMAIL_APP_PASSWORD = "tayc gzvm bkqz zlxu"

PIP = 0.0001


def pips(val):
    return val / PIP


def get_csv_files():
    return sorted(glob.glob(os.path.join(RESULTS_DIR, CSV_PATTERN)))


def get_dp_pair_files():
    return sorted(glob.glob(os.path.join(RESULTS_DIR, f"dp_pair_v{VERSION}_*.csv")))


def analyze_dp_pair(files):
    if not files:
        return None
    per_loop = []
    for f in files:
        df = pd.read_csv(f)
        if 'dp_pair' not in df.columns:
            continue
        loop = {'file': os.path.basename(f)}
        for pair_type in ['bull', 'bear']:
            sub = df[df.pair_type == pair_type]
            if len(sub) == 0:
                continue
            loop[pair_type] = {'n': len(sub), 'avg': sub['dp_pair'].mean()}
        per_loop.append(loop)
    return per_loop if per_loop else None


def analyze(files):
    if not files:
        return None

    all_dfs  = []
    per_file = []

    for i, f in enumerate(files, 1):
        df = pd.read_csv(f)
        if 'pnl' not in df.columns:
            with open(f) as _tmp:
                ncols = len(next(_tmp).split(','))
            if ncols >= 8:
                names = ['side', 'real', 'entry', 'exit', 'pnl', 'entry_transition', 'bull_pairs', 'bear_pairs']
            elif ncols >= 7:
                names = ['side', 'entry', 'exit', 'pnl', 'entry_transition', 'bull_pairs', 'bear_pairs']
            else:
                names = ['side', 'entry', 'exit', 'pnl', 'pnl_pct', 'entry_transition']
            df = pd.read_csv(f, names=names, header=None, index_col=False)
        all_dfs.append(df)

        n        = len(df)
        w        = (df.pnl > 0).sum()
        l        = (df.pnl < 0).sum()
        flat     = (df.pnl == 0).sum()
        wr       = w / n * 100 if n > 0 else 0
        loop_pnl = df.pnl.sum()
        avg      = df.pnl.mean()
        best     = df.pnl.max()
        worst    = df.pnl.min()

        header = (
            f"  File {i:>3}: {os.path.basename(f)}\n"
            f"           {n:>3} trades | W={w:>3} L={l:>3} F={flat:>2} | "
            f"win={wr:>5.1f}% | PnL={pips(loop_pnl):>+7.1f} pips | "
            f"avg={pips(avg):>+5.2f} | best={pips(best):>+5.1f} | worst={pips(worst):>+5.1f}"
        )
        trades = []
        for _, row in df.iterrows():
            result = 'W' if row.pnl > 0 else ('F' if row.pnl == 0 else 'L')
            pairs = ""
            if 'bull_pairs' in df.columns:
                pairs = f" bull={int(row.bull_pairs)} bear={int(row.bear_pairs)}"
            trades.append(
                f"    {result} {row.side:<5} | entry={row.entry:.4f} exit={row.exit:.4f} | "
                f"PnL={pips(row.pnl):>+6.1f} pips | {row.entry_transition}{pairs}"
            )
        per_file.append((header, trades))

    combined  = pd.concat(all_dfs)
    n         = len(combined)
    w         = (combined.pnl > 0).sum()
    l         = (combined.pnl < 0).sum()
    flat      = (combined.pnl == 0).sum()
    wr        = w / n * 100 if n > 0 else 0
    total_pnl = combined.pnl.sum()
    avg       = combined.pnl.mean()
    best      = combined.pnl.max()
    worst     = combined.pnl.min()

    long_df   = combined[combined.side == "LONG"]
    short_df  = combined[combined.side == "SHORT"]
    long_pnl  = long_df.pnl.sum()  if len(long_df)  > 0 else 0
    short_pnl = short_df.pnl.sum() if len(short_df) > 0 else 0
    long_wr   = (long_df.pnl > 0).sum()  / len(long_df)  * 100 if len(long_df)  > 0 else 0
    short_wr  = (short_df.pnl > 0).sum() / len(short_df) * 100 if len(short_df) > 0 else 0
    long_avg  = long_df.pnl.mean()  if len(long_df)  > 0 else 0
    short_avg = short_df.pnl.mean() if len(short_df) > 0 else 0

    dp_pair = analyze_dp_pair(get_dp_pair_files())
    dp_section = "\nΔP_PAIR PER LOOP\n"
    if dp_pair:
        for i, loop in enumerate(dp_pair, 1):
            parts = []
            for pair_type in ['bull', 'bear']:
                if pair_type in loop:
                    s = loop[pair_type]
                    parts.append(f"{pair_type}={s['avg']:>+.4f} (n={s['n']})")
            dp_section += f"  Loop {i:>3}: {' | '.join(parts)}\n"
    else:
        dp_section += "  no data yet\n"

    header = f"""SKA Trading Bot v{VERSION} — Report ({len(files)} files)
{'=' * 50}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

SUMMARY
  Total trades: {n}
  Winners: {w} | Losers: {l} | Flat: {flat}
  Win rate: {wr:.1f}%
  Total PnL:    {pips(total_pnl):>+9.1f} pips
  Avg PnL/trade:{pips(avg):>+9.2f} pips
  Best trade:   {pips(best):>+9.1f} pips
  Worst trade:  {pips(worst):>+9.1f} pips

BY SIDE
  LONG:  {len(long_df):>4} trades | PnL={pips(long_pnl):>+8.1f} pips | avg={pips(long_avg):>+5.2f} pips | win_rate={long_wr:.1f}%
  SHORT: {len(short_df):>4} trades | PnL={pips(short_pnl):>+8.1f} pips | avg={pips(short_avg):>+5.2f} pips | win_rate={short_wr:.1f}%
{dp_section}
PER FILE
"""
    summary  = header + '\n'.join(h for h, _ in per_file)
    detailed = header + '\n'.join(h + '\n' + '\n'.join(t) for h, t in per_file)

    return summary, detailed


def send_email(subject, body):
    password = GMAIL_APP_PASSWORD
    if not password or not EMAIL_FROM or not EMAIL_TO:
        print("WARNING: email not configured — printing report to console.")
        print(body)
        return False

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"]    = EMAIL_FROM
    msg["To"]      = EMAIL_TO

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_FROM, password)
            server.sendmail(EMAIL_FROM, EMAIL_TO, msg.as_string())
        print(f"Email sent to {EMAIL_TO}")
        return True
    except Exception as e:
        print(f"Email failed: {e}")
        print(body)
        return False


def main():
    print(f"SKA Bot Monitor started — watching: {os.path.abspath(RESULTS_DIR)}")
    last_count = 0

    while True:
        files         = get_csv_files()
        current_count = len(files)

        if current_count > last_count:
            result = analyze(files)
            if result:
                summary, detailed = result
                subject  = f"SKA Bot Report — {current_count} files — PnL={summary.split('Total PnL:')[1].split()[0]} pips"
                txt_path = os.path.join(RESULTS_DIR, f"ska_report_v{VERSION}_{current_count}_files.txt")
                with open(txt_path, 'w') as f:
                    f.write(detailed)
                print(f"Report saved: {txt_path}")
                send_email(subject, detailed)
                last_count = current_count

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
