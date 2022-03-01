#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

/* FUNCTIONS */

void
llinsert(ll **lln, int lnum)
{
	if((*lln)==NULL){
		(*lln) = malloc(sizeof(ll));
		(*lln)->data = lnum;
		(*lln)->next = NULL;
		return;
	}
	llinsert(&((*lln)->next),lnum);
	return;
}


void
bstinsert(bst **bstn, char word[], int lnum){
	int len = strlen(word);
	if(!len)
		return;
	int index = 0;
	if((*bstn)==NULL){
		(*bstn) = malloc(sizeof(bst));
		/* Change the case of the word*/
		strncpy((*bstn)->word,word,21);
		(*bstn)->left = NULL;
		(*bstn)->right = NULL;
		llinsert(&((*bstn)->linenum), lnum);
		return;
	}
	if(strcasecmp((*bstn)->word,word)==0){
		llinsert(&((*bstn)->linenum), lnum);
		return;
	}
	if(strcasecmp((*bstn)->word,word)>0){
		bstinsert(&((*bstn)->left), word, lnum);
		return;
	}
	bstinsert(&((*bstn)->right), word, lnum);
	return;
}
void
llprint(ll *lln){
	if(lln==NULL)
		return;
	llprint(lln->next);
	printf(" %d, ",lln->data);
	return;
}
void
bstprint(bst *bstn){
	if(bstn == NULL)
		return;
	if(bstn->left==NULL){
		printf(" %s:",bstn->word);
		llprint(bstn->linenum);
		printf("\n");
		if(bstn->right==NULL)
			return;
		bstprint(bstn->right);
		return;
	}
	bstprint(bstn->left);
	printf(" %s:",bstn->word);
	llprint(bstn->linenum);
	printf("\n");
	bstprint(bstn->right);
	return;
}

void
freell(ll *lln){
	if(lln==NULL)
		return;
	freell(lln->next);
	free(lln);
	return;
}

void
freebst(bst *bstn){
	if(bstn==NULL)
		return;
	if(bstn->left==NULL){
		if(bstn->right==NULL){
			freell(bstn->linenum);
			free(bstn);
			return;
		}
		freebst(bstn->right);
		free(bstn);
		return;
	}
	freebst(bstn->left);
	freebst(bstn->right);
	free(bstn);
	return;
}

int
main( int argc, char *argv[]){
	bst *one = NULL;
	bstinsert(&one,"hello",1);
	bstinsert(&one,"hello",1);
	bstinsert(&one,"ello",1);
	bstinsert(&one,"hlo",1);
	bstprint(one);
	freebst(one);
	return 0;
}

