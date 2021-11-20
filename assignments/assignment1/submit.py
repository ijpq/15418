#!/usr/bin/python

# This program will submit a solution to the assignment by inserting a
# file into a directory that has restricted access.

# This program should only be used by students who cannot submit
# solutions via Autolab (e.g., those who are still on the waitlist).

import shutil
import sys
import getopt
import glob

submitDirectory = "/afs/cs.cmu.edu/academic/class/15418-f19/public/asst1-handin"

def usage(name):
    print "Usage: %s -u USER [-h]" % name
    print "       -u USER   Specify Andrew ID of student"
    print "       -h        Print this message"

def submit(id):
    global submitDirectory
    version = 1
    prefix = submitDirectory + "/handin-" + id + "-v"
    template = prefix + "*.tar"
    flist = glob.glob(template)
    for fname in flist:
        pos = len(prefix)
        digits = ""
        char = fname[pos]
        while char.isdigit():
            digits += char
            pos += 1
            char = fname[pos]
        nversion = int(digits)
        if nversion >= version:
            version = nversion + 1
    destName = prefix + str(version) + ".tar"
    try:
        shutil.copyfile("handin.tar", destName)
    except Exception as e:
        print "FAILED to copy handin.tar to destination '%s'.  (%s)" % (destName, e)
        sys.exit(1)
    print "SUCCESSFULLY copied handin.tar to destination '%s'." % destName

def run(name, args):
    id = ""
    optlist, args = getopt.getopt(args, "hu:")
    for opt, val in optlist:
        if opt == '-h':
            usage(name)
            sys.exit(0)
        elif opt == '-u':
            id = val
        else:
            print "Unknown option '%s'" % opt
            usage(name)
            sys.exit(0)

    if id == "":
        print "You must provide your Andrew Id"
        usage(name)
        sys.exit(0)
    submit(id)

if __name__ == "__main__":
    run(sys.argv[0], sys.argv[1:])

    
