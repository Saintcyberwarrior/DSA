#include <stdio.h>

int main(){
	int a;
	char c;
	while(a = getc(stdin)){
		putc(a, stdout);
		if(a=='$')
			return 1;
	}
	return 0;
}
