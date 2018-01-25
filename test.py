#!/usr/bin/env python3

import time

import lib.cli as cli

print(cli.args.get('v'))

start = cli.log.now()
time.sleep(1)
cli.log.doneIn(start)
