CC=gcc
CFLAGS=-O3 -std=gnu99
LDFLAGS=
LIBS=-lm

all: axpy

axpy: axpy.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS) $(LIBS)

clean:
	$(RM) axpy
