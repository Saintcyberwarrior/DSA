#include <stdio.h>
#include <stdlib.h>
#include <string.h>
/*
	USAGE::
		bst_insert(pointer to node, word to be inserted,
				line number of the word);



*/
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
	struct ll *linenum;
	struct bst *left, *right;
};
void
ll_insert(ll *lnode, int lnum)
{
	if(!(*lnode)){
		lnode = malloc(sizeof(ll));


void
bst_insert(bst *bstnode, char word[], int lnum){
	int len = strlen(word);
	if(!len)
		return;
	int index = 0;
	if(!(*bstnode)){
		bstnode = malloc(sizeof(bst));
		strncpy(bstnode->word,word);
		bstnode->left = NULL;
		bstnode->right = NULL;

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

