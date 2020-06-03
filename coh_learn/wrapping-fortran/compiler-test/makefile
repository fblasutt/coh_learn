MAINFILE = HA2
LIBS = /usr/local/include/toolbox.o
CFLAGS = -O2 -I/usr/local/include/
CC = /opt/local/bin/gfortran-mp-4.9
TARGET = Build/prog

all: $(TARGET)
	clear
	./$(TARGET)
	-rm -rf Build
	-rm -f *.mod

$(TARGET): Build/$(MAINFILE).o
	$(CC) -O2 $(LIBS) Build/$(MAINFILE).o -o $(TARGET)

Build/$(MAINFILE).o: $(MAINFILE).f90
	mkdir -p Build
	$(CC) $(CFLAGS) -c $(MAINFILE).f90 -o Build/$(MAINFILE).o


clean:
	-rm -rf Build
	-rm -f *.mod