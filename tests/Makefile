CC=g++
CFLAGS=-Wall -c -g

all: testbase

testbase: testbase.o base.o exceptions.o
	$(CC) testbase.o base.o exceptions.o -o testbase -lcppunit

testbase.o: testbase.cpp
	$(CC) $(CFLAGS) testbase.cpp

base.o: ../src/simpl/base.cpp
	$(CC) $(CFLAGS) ../src/simpl/base.cpp

exceptions.o: ../src/simpl/exceptions.cpp
	$(CC) $(CFLAGS) ../src/simpl/exceptions.cpp

clean:
	rm -rf *.o testbase
