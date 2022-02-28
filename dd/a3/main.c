#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct
ll
{
	int data;
	struct ll *next;
};
typedef struct
bst
{
	int index;
	char word[21];
	struct ll *lineNum;
	struct bst *left, *right;
};

int
main(int argc, char *argv[]){
	if(argc==1){
		printf(" USAGE:\n");
		printf(" \t%s <filename>\n",argv[0]);
		printf(" filename is the txt file\n Aborting...\n");
		return 0;
	}

	FILE *in;

	in = fopen(argv[1],"r");

	if(!in){
		fprintf(stderr," Cannot open file!!\n");
		return 1;
	}

	int ch;
	char buff[30];
	int buffCount = 0;


	while((ch = fgetc(in)) && ch!=EOF){
		if(ch == ' ' || ch =='\n' || ch == '\t'){
			if(strlen(buff)){
				buff(
			buffCount = 0;
			continue;
		}
		else{


		fputc(ch,stdout);
	}

	fclose(in);
	return 0;
}

