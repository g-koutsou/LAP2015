CC=gcc
CFLAGS=-O3 -std=gnu99 -fopenmp
LDFLAGS=
LIBS=-lm -fopenmp

all: axpy

axpy: axpy.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS) $(LIBS)

clean:
	$(RM) axpy
