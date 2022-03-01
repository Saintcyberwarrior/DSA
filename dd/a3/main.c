#include <stdio.h>
#include <stdlib.h>
#include <string.h>
/*
	USAGE::
		bst_insert(pointer to node, word to be inserted,
				line number of the word);



*/
typedef struct
llnode
{
	int data;
	struct llnode *next;
} ll;
typedef struct
bstnode
{
	int index;
	char word[21];
	struct llnode *linenum;
	struct bstnode *left, *right;
} bst;
void
ll_insert(ll *lln, int lnum)
{
	if(lln==NULL){
		lln = malloc(sizeof(ll));
		lln->data = lnum;
		lln->next = NULL;
	}
	ll_insert(lln->next,lnum);
}


void
bst_insert(bst *bstn, char word[], int lnum){
	int len = strlen(word);
	if(!len)
		return;
	int index = 0;
	if(bstn==NULL){
		bstn = malloc(sizeof(bst));
		strncpy(bstn->word,word,21);
		bstn->left = NULL;
		bstn->right = NULL;
		ll_insert(bstn->linenum, lnum);
	}
}

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
		fprintf(stderr," Cannot open file!!!\n");
		return 1;
	}

	int ch;
	char buff[30];
	int buffCount = 0;


	while((ch = fgetc(in)) && ch!=EOF){
		if(ch == ' ' || ch =='\n' || ch == '\t'){
			if(strlen(buff)){
				buff[buffCount] = '\0';
				//
				//
				//
				// Add buff to the tree
				//
				//
				printf("%s",buff);
			}
			printf("\n");

			buffCount = 0;
			continue;
		}
		else{
			buff[buffCount] = ch;
			buffCount ++;
		}


	}

	fclose(in);
	return 0;
}

