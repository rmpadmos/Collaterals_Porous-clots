#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
Transcript - direct print output to a file, in addition to terminal.

Usage:
    import transcript
    transcript.start('logfile.log')
    print("inside file")
    transcript.stop()
    print("outside file")
"""

import sys


class Transcript(object):
    def __init__(self, filename):
        """
        Initiate object

        Parameters
        ----------
        filename : str
            log file name
        """
        self.terminal = sys.stdout
        self.logfile = open(filename, "a")

    def write(self, message):
        """
        Write message to console and logfile.

        Parameters
        ----------
        message : str
            message to print
        """
        self.terminal.write(message)
        self.logfile.write(message)

    def flush(self):
        """
        This flush method is needed for python 3 compatibility.
        This handles the flush command by doing nothing.
        You might want to specify some extra behavior here.
        """
        pass


def start(filename):
    """
    Start transcript, appending print output to given filename.

    Parameters
    ----------
    filename : str
        logfile name
    """
    sys.stdout = Transcript(filename)


def stop():
    """
    Stop transcript and return print functionality to normal.
    """
    sys.stdout.logfile.close()
    sys.stdout = sys.stdout.terminal
