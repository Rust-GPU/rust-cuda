#!/bin/bash
set -e

# This script is used by cargo to run the test runner with the 
# specified arguments.
#
# We need this script because the cwd that cargo runs the runner in changes
# depending on crate.

cd "$GITHUB_WORKSPACE/.ci/"
echo "In glue script with cwd: $(pwd)"
echo "Got args: $@"
