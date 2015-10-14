CC=gcc
CFLAGS=-O3 -D_GNU_SOURCE -std=c99 -fopenmp
LDFLAGS=
LIBS=-lm -fopenmp
all: lapl 

lapl: lapl.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS) $(LIBS)

clean:
	$(RM) lapl
