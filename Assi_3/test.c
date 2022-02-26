#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

struct bst_node * mk_tree_node(struct bst_node * l, struct bst_node * r, int new_index, char new_word, struct lst_node* new_lst);
struct bst_node* bst_add(struct bst_node *a, int new_index, char new_word, struct lst_node* new_lst);
void print(lst_node * temp);
void tree_inorder_traverse(struct bst_node* a);
void frre_lst(struct lst_node *a);
void bpush(struct lst_node** a, int new_data);


struct bst_node * mk_tree_node(struct bst_node * l, struct bst_node * r, int new_index, char new_word, struct lst_node* new_lst){
	struct bst_node * tmp_node = (struct bst_node*)malloc(sizeof (struct bst_node ));
	tmp_node->left = l;
	tmp_node->right = r;
	tmp_node->index = new_index;
	tmp_node->word = new_word;
	tmp_node->list = new_list;

	return tmp_node;
}
struct bst_node* bst_add(struct bst_node *a, int new_index, char new_word, struct lst_node* new_lst){
	if (a == NULL)
		a = mk_tree_node(NULL, NULL, new_index, new_word, new_lst);
	else if (a->index < new_index)
		a->right = bst_add(a->right, new_index, new_word, new_lst);
	else if (a->index > new_index)
		a->left= bst_add(a->left, new_index, new_word, new_lst);
	return a;
}
void print(lst_node * temp){
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
	struct lst_node* new_node = (struct lst_node*)malloc(sizeof(struct lst_node))
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

void frre_lst(struct lst_node *a){
	struct lst_node* temp = a;
	while(temp != NULL && a->next != NULL){
		temp = a->next;
		free(a);
		a = temp;
	}
}

