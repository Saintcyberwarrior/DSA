#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define EndOfWord word[i][j] == ',' || word[i][j] == '\n' ||word[i][j] == ';'|| word[i][j] == ' ' \
			      ||word[i][j] == '.'|| word[i][j] =='!' || word[i][j] == '?' || word[i][j] == '''|| word[i][j] == '"'

struct lst_node{
	int data;
	struct lst_node* next;
};
struct bst_node {
	char word[21];
	int index;
	struct bst_node *left, *right;
	struct lst_node* list;
};

struct bst_node * mk_tree_node(struct bst_node * l, struct bst_node * r, int new_index, char new_word[21], struct lst_node* new_lst);
struct bst_node* bst_add(struct bst_node *a, int new_index, char new_word[21], struct lst_node* new_lst);
void print(struct lst_node * temp);
void tree_inorder_traverse(struct bst_node* a);
void bpush(struct lst_node** a, int new_data);
int len_lst(struct lst_node*temp);
void frre_lst(struct lst_node *a);
void frre_bst(struct bst_node *a);
void swap(char *one, char *two);


int main(int argc, char *argv[]){
	FILE *fp;
	fp = fopen("", "r");
	if (fp == NULL){
		fprintf(stderr, "Cannot open File\n");
		exit(1);
	}


}

struct bst_node * mk_tree_node(struct bst_node * l, struct bst_node * r, int new_index, char new_word[21], struct lst_node* new_lst){
	struct bst_node * tmp_node = (struct bst_node*)malloc(sizeof (struct bst_node ));
	tmp_node->left = l;
	tmp_node->right = r;
	tmp_node->index = new_index;
	tmp_node->word = new_word;
	tmp_node->list = new_lst;

	return tmp_node;
}
struct bst_node* bst_add(struct bst_node *a, int new_index, char new_word[21], struct lst_node* new_lst){
	if (a == NULL)
		a = mk_tree_node(NULL, NULL, new_index, new_word, new_lst);
	else if (a->index < new_index)
		a->right = bst_add(a->right, new_index, new_word, new_lst);
	else if (a->index > new_index)
		a->left= bst_add(a->left, new_index, new_word, new_lst);
	return a;
}
void print(struct lst_node * temp){
	while(temp){
		if(!(temp->next)){
			printf("%d\n", temp->data);
			return ;
		}
		printf("%d,", temp->data);
		temp = temp->next;
	}
}


void tree_inorder_traverse(struct bst_node* a){
	tree_inorder_traverse(a->left);
	printf("%s\t", a->word);
	print(a->list);
}

void bpush(struct lst_node** a, int new_data){
	struct lst_node* new_node = (struct lst_node*)malloc(sizeof(struct lst_node));
	if(!new_node){
		fprintf(stderr, "Cannot Allocate Memory\n");
		exit(1);
	}

	new_node->data = new_data;
	new_node->next = NULL;
	if((*a) == NULL){
		(*a) = new_node;
		return;
	}
	struct lst_node* temp = (*a);
	while(temp->next){
		temp = temp->next;
	}
	temp->next = new_node;
}

int len_lst(struct lst_node*temp){
	int count = 1;
//	struct lst_node* temp = a;
	while(temp != NULL){
		if(temp->next == NULL)
			return count;
		temp = temp->next;
		count++;
	}
}

void frre_lst(struct lst_node *a){
	struct lst_node* temp = a;
	while(temp != NULL && a->next != NULL){
		temp = a->next;
		free(a);
		a = temp;
	}
}

void frre_bst(struct bst_node *a){
	if(a!=NULL){
		frre_bst(a->left);
		frre_bst(a->right);
		free(a);
	}
}

void swap(char *one, char *two){
	char three[20];
	strcpy(three, one);
	strcpy(one, two);
	strcpy(two, three);
}

