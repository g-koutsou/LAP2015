CC=gcc
CFLAGS=-O3 -D_GNU_SOURCE -std=c99 -fopenmp
LDFLAGS=
LIBS=-lm -fopenmp
N?=1
all: mxam

mxam: mxam.c
	$(CC) -DN=$(N) $(CFLAGS) $< -o $@ $(LDFLAGS) $(LIBS)

clean:
	$(RM) mxam
