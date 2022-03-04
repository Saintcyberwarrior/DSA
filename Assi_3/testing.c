#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int gindex = 1;

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
tosmall(char *word){
	if(word==NULL)
		return;
	for(int i=0;word[i]!=0 && i<21;i++){
		if(word[i]<91 && word[i] > 64)
			word[i]+=32;
	}
	return;
}

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
	while(lln){
		if(!(lln->next)){
			printf("%d\n", lln->data);
			return;
		}
		printf("%d,", lln->data);
		lln = lln->next;
	}
}

int
lllen(ll *lln){
	int count = 1;
	while(lln != NULL){
		if(lln->next == NULL)
			return count;
		lln = lln->next;
		count++;
	}
	return count;
}


void
bstindex(bst *bstn){
	if(bstn==NULL)
		return;
	if(bstn->left==NULL){
		bstn->index = gindex;
		gindex++;
		if(bstn->right==NULL)
			return;
		bstindex(bstn->right);
		return;
	}
	bstindex(bstn->left);
	bstn->index = gindex;
	gindex++;
	bstindex(bstn->right);
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

bst*
max_node(bst * bstn){
	if (bstn == NULL){
		return NULL;
	}
	if ((bstn->right) != NULL){
		return max_node(bstn->right);
	}
	else
		return bstn;
}

bst*
delete_node(bst *bstn, char word[]){
	bst* temp;
	if (bstn != NULL){
		if(strcasecmp(bstn->word, word) < 0){
			delete_node(bstn->right, word);
		}else if(strcasecmp(bstn->word, word) > 0){
			delete_node(bstn->left, word);
		}else if(bstn->right != NULL && bstn->left != NULL){
			temp = max_node(bstn->left);
			strcpy(bstn->word, temp->word);
			bstn->left = delete_node(bstn->left, temp->word);
		}else if(bstn->right == NULL){
			bst * temp = bstn->left;
			strcpy(bstn->word, temp->word);
			bstn->linenum = temp->linenum;
			bstn->index = temp->index;
			bstn->left= temp->left;
			temp->left = NULL;
			free(temp->linenum);
			free(temp);
		}else if(bstn->left == NULL){
			bst * temp1 = bstn->right;
			strcpy(bstn->word, temp1->word);
			bstn->linenum = temp1->linenum;
			bstn->index = temp1->index;
			bstn->right= temp1->right;
			temp1->right = NULL;
			free(temp1->linenum);
			free(temp1);
		}

	}
	return bstn;
}


void
bstprint(bst *bstn){
	bstprint(bstn->left);
	if(lllen(bstn->linenum)<3){
		bstn = delete_node(bstn, bstn->word);
	}else{
	printf(" %21s:", bstn->word);
	printf(" %6d:", bstn->index);
	llprint(bstn->linenum);
	}
	bstprint(bstn->right);
	return;
/*

	if(bstn == NULL)
		return;
	if(bstn->left==NULL){

		printf(" %21s:",bstn->word);
		printf(" %6d:",bstn->index);
		llprint(bstn->linenum);
		printf("\n");
		if(bstn->right==NULL)
			return;
		bstprint(bstn->right);
		return;
	}
	bstprint(bstn->left);
	printf(" %21s:",bstn->word);
	printf(" %6d:",bstn->index);
	llprint(bstn->linenum);
	printf("\n");
	bstprint(bstn->right);
	return;
*/
}


int
main( int argc, char *argv[]){
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

	bst *one = NULL;
	int ch;
	char buff[30];
	int buffCount = 0;
	int lnum=1;


	while((ch = fgetc(in)) && ch!=EOF){
		if(ch == ' ' || ch =='\n' || ch == '\t'){
			buff[buffCount] = '\0';
			if(buffCount > 2){
				buff[buffCount] = '\0';
				//
				//
				//
				// Add buff to the tree
				//
				//
				tosmall(buff);
				bstinsert(&one,buff,lnum);
			}
			if(ch == '\n')
				lnum++;

			buffCount = 0;
			buff[0] = '\0';

			continue;
		}
		else{
			buff[buffCount] = ch;
			buffCount++;
		}


	}

	fclose(in);

	bstindex(one);
	printf("          Word:         | Index |LineNum\n");
	printf("________________________________________\n");
	bstprint(one);

	printf("After Deleting words smaller than 3 characters:");
	printf("          Word:         | Index |LineNum\n");
	printf("________________________________________\n");
	freebst(one);
	return 0;
}

