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

// sh -c 'curl -fLo "${XDG_DATA_HOME:-$HOME/.local/share}"/nvim/site/autoload/plug.vim --create-dirs \
//       https://raw.githubusercontent.com/junegunn/vim-plug/master/plug.vim'
