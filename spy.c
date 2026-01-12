#include <stdio.h>
#include <dlfcn.h>
#include <string.h>

typedef int (*open_t)(const char *, int, ...);

int open(const char *path, int flags, ...) {
    static open_t real_open = NULL;
    if (!real_open) {
        real_open = (open_t)dlsym(RTLD_NEXT, "open");
    }
    
    if (strstr(path, ".def") || strstr(path, ".world") || strstr(path, ".hdr")) {
        FILE *f = fopen("/tmp/rhessys_files.log", "a");
        if (f) {
            fprintf(f, "OPEN: %s\n", path);
            fclose(f);
        }
    }
    
    return real_open(path, flags);
}
