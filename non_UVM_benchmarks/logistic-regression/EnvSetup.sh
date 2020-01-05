#!/bin/bash

export PGI=/opt/pgi;
export PATH=$PGI/linux86-64/17.4/bin:$PATH;
export MANPATH=$MANPATH:$PGI/linux86-64/17.4/man;
export LM_LICENSE_FILE=$LM_LICENSE_FILE:/opt/pgi/license.dat;

# /opt/pgi/linux86-64/17.4/bin/makelocalrc  -x /opt/pgi/linux86-64/17.4 -net /usr/pgi/shared/17.4
