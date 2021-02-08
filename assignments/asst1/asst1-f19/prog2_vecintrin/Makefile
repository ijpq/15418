all: vrun

logger.o: logger.cpp logger.h CMU418intrin.h CMU418intrin.cpp
	g++ -c logger.cpp

CMU418intrin.o: CMU418intrin.cpp CMU418intrin.h logger.cpp logger.h
	g++ -c CMU418intrin.cpp

vrun: CMU418intrin.o logger.o main.cpp functions.cpp
	g++ -I../common logger.o CMU418intrin.o main.cpp functions.cpp -o vrun

clean:
	rm -f *.o *~ vrun
