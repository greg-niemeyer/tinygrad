#!/usr/bin/env python3
# compare kernels created by HEAD against master
import difflib, pickle, multiprocessing, os
from tinygrad.codegen.kernel import Kernel
from tinygrad.helpers import Context, ContextVar, colored, db_connection, VERSION, getenv, tqdm
from process_replay_test_generator import generate_test_script

from time import time
tests = []

page_size = 100
table_name = f"process_replay_{getenv('GITHUB_RUN_ID', 'HEAD')}_{VERSION}"

def process_replay(offset:int):
  ASSERT_PROCESS_REPLAY = (k:="[run_process_replay]") in os.getenv("COMMIT_MESSAGE", k) or k in os.getenv("PR_TITLE", k)
  conn = db_connection()
  cur = conn.cursor()
  cur.execute(f"SELECT val FROM '{table_name}' LIMIT ? OFFSET ?", (page_size, offset))
  for row in cur.fetchall():
    ast, opts, applied_opts, compare_uops, name, compare_src, ctx = pickle.loads(row[0])
    with Context(**{k:v for k,v in ctx.items() if k in ContextVar._cache}):
      # try linearize
      try:
        k = Kernel(ast, opts=opts)
        for opt in applied_opts: k.apply_opt(opt)
        good_src = k.opts.render(name, k.linearize().uops)
        good_uops = str(k.uops)
      except Exception as e:
        print("FAILED TO RECREATE KERNEL")
        print(ast)
        print(applied_opts)
        print(e)
        if ASSERT_PROCESS_REPLAY: raise e
        continue
      # try compare
      try: assert compare_src == good_src
      except AssertionError as e:
        print("PROCESS REPLAY DETECTED CHANGE")
        print(ast)
        print(applied_opts)
        tests.append((ast, applied_opts))
        diff = list(difflib.unified_diff(good_uops.splitlines(), compare_uops.splitlines()))
        for line in diff:
          print(colored(line, "red" if line.startswith("-") else "green" if line.startswith("+") else None))
        if getenv("ASSERT_PROCESS_REPLAY", 1): raise e
        diff = list(difflib.unified_diff(good_src.splitlines(), compare_src.splitlines()))
        for line in diff:
          print(colored(line, "red" if line.startswith("-") else "green" if line.startswith("+") else None))
        if ASSERT_PROCESS_REPLAY: raise e
  if len(tests): generate_test_script(tests, f"process_replay_{int(time()):08d}.py")
  conn.commit()
  cur.close()

if __name__ == "__main__":
  conn = db_connection()
  cur = conn.cursor()
  row_count = cur.execute(f"select count(*) from '{table_name}'").fetchone()[0]
  conn.commit()
  cur.close()
  offsets = range(0, row_count, page_size)
  with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool: list(tqdm(pool.imap(process_replay, offsets), total=len(offsets)))