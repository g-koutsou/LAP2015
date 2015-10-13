CC=gcc
CFLAGS=-O3 -std=gnu99 -fopenmp
LDFLAGS=
LIBS=-lm -fopenmp
N?=1
all: mxam

mxam: mxam.c
	$(CC) -DN=$(N) $(CFLAGS) $< -o $@ $(LDFLAGS) $(LIBS)

clean:
	$(RM) mxam
