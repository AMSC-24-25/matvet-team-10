# Compiler and related settings
CXX = mpic++   # Default compiler

# Compiler flags
DEBUG = no
ifeq ($(DEBUG), yes)
    STDFLAGS = -std=c++17 -g -O0 -Wall -Wextra -pedantic
else
    STDFLAGS = -std=c++17 -O2 -Wall -Wextra -pedantic
endif

# Preprocessor flags
CPPFLAGS = -I.   # Include current directory for headers

# Linker flags
LDFLAGS = 
LDLIBS = 

# Dependency file
DEPEND = .depend
