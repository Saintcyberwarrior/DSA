#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define EndOfWord word[i] == ',' || word[i] == '\n' ||word[i] == ';'|| word[i] == ' ' \
			      ||word[i] == '.'|| word[i] =='!' || word[i] == '?'
//			      || word[i][j] == '''|| word[i][j] == '"'

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

struct bst_node * mk_tree_node(struct bst_node * l, struct bst_node * r, int new_index, char new_word[21], int new_line_no);
struct bst_node* bst_add(struct bst_node *a, int new_index, char new_word[21], int new_line_no);
void print(struct lst_node * temp);
void tree_inorder_traverse(struct bst_node* a);
void bpush(struct lst_node** a, int new_data);
int len_lst(struct lst_node*temp);
void frre_lst(struct lst_node *a);
void frre_bst(struct bst_node *a);
void swap(char *one, char *two);
void lst_insert(struct lst_node* a, int new_data);
struct bst_node* reader(FILE *file);

int main(int argc, char *argv[]){
	FILE *fp;
	fp = fopen("./data.txt", "r");
	struct bst_node* Ultimate_tree = NULL;

	Ultimate_tree = reader(fp);
    tree_inorder_traverse(Ultimate_tree);
	return 0;
}

struct bst_node * mk_tree_node(struct bst_node * l, struct bst_node * r, int new_index, char new_word[21], int new_line_no)
{
	struct bst_node * tmp_node = (struct bst_node*)malloc(sizeof (struct bst_node ));
	tmp_node->left = l;
	tmp_node->right = r;
	tmp_node->index = new_index;
	strncpy(tmp_node->word, new_word, 21);
	lst_insert(tmp_node->list, new_line_no);
	return tmp_node;
}
struct bst_node* bst_add(struct bst_node *a, int new_index, char new_word[21], int new_line_no)
{
	if (a == NULL)
		a = mk_tree_node(NULL, NULL, new_index, new_word, new_line_no);
	else if (strcasecmp(a->word, new_word)<0)
		a->right = bst_add(a->right, new_index, new_word, new_line_no);
	else if (strcasecmp(a->word, new_word)>0)
		a->left= bst_add(a->left, new_index, new_word, new_line_no);
	else if (strcasecmp(a->word, new_word)==0)
		lst_insert(a->list, new_line_no);
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
		frre_lst(a->list);
	}
}

void swap(char *one, char *two){
	char three[20];
	strcpy(three, one);
	strcpy(one, two);
	strcpy(two, three);
}

void lst_insert(struct lst_node* a, int new_data){
	if(a==NULL){
		a = malloc(sizeof(struct lst_node));
		a->data = new_data;
		a->next = NULL;

	}
	lst_insert(a->next, new_data);
}
/*
void update_node(struct bst_node *a, int new_index, char word[21], int line_no){
	struct bst_node *b = NULL;
	b = search_node(a, word);
	b->index = new_index;
	lst_insert(b->list, line_no);
}

struct bst_node* search_node(struct bst_node *a, char word[]){
	if(strcasecmp(a->word, word) == 0){
		return a;
	}else if(strcasecmp(a->word, word) < 0){
		search_node(a->left, word);
	}else if(strcasecmp(a->word, word) > 0){
		search_node(a->right, word);
	}else if(a->word = NULL){
		return NULL;
	}
}*/
/*struct bst_node* reader(FILE *file){
	if(!file){
		fprintf(stderr, "Couldn't open file\n");
		exit(1);
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define EndOfWord word[i] == ',' || word[i] == '\n' ||word[i] == ';'|| word[i] == ' ' \
			      ||word[i] == '.'|| word[i] =='!' || word[i] == '?'
			      || word[i] == '''|| word[i] == '"'

*/
/*
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

struct bst_node * mk_tree_node(struct bst_node * l, struct bst_node * r, int new_index, char new_word[21], int new_line_no);
struct bst_node* bst_add(struct bst_node *a, int new_index, char new_word[21], int new_line_no);
void print(struct lst_node * temp);
void tree_inorder_traverse(struct bst_node* a);
void bpush(struct lst_node** a, int new_data);
int len_lst(struct lst_node*temp);
void frre_lst(struct lst_node *a);
void frre_bst(struct bst_node *a);
void swap(char *one, char *two);
void lst_insert(struct lst_node* a, int new_data);
struct bst_node* reader(FILE *file);

int main(int argc, char *argv[]){
	FILE *fp;
	fp = fopen("./data.txt", "r");
	struct bst_node* Ultimate_tree = NULL;

	Ultimate_tree = reader(fp);
	tree_inorder_traverse(Ultimate_tree);
	return 0;
}

struct bst_node * mk_tree_node(struct bst_node * l, struct bst_node * r, int new_index, char new_word[21], int new_line_no)
{
	struct bst_node * tmp_node = (struct bst_node*)malloc(sizeof (struct bst_node ));
	tmp_node->left = l;
	tmp_node->right = r;
	tmp_node->index = new_index;
	strcnpy(tmp_node->word, new_word, 21);
	lst_insert(tmp_node->list, new_line_no);
	return tmp_node;
}
struct bst_node* bst_add(struct bst_node *a, int new_index, char new_word[21], int new_line_no)
{
	if (a == NULL)
		a = mk_tree_node(NULL, NULL, new_index, new_word, new_line_no);
	else if (strcasecmp(a->word, new_word)<0)
		a->right = bst_add(a->right, new_index, new_word, new_line_no);
	else if (strcasecmp(a->word, new_word)>0)
		a->left= bst_add(a->left, new_index, new_word, new_line_no);
	else if (strcasecmp(a->word, new_word)==0)
		lst_insert(a->list, new_line_no);
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
		frre_lst(a->list);
	}
}

void swap(char *one, char *two){
	char three[20];
	strcpy(three, one);
	strcpy(one, two);
	strcpy(two, three);
}

void lst_insert(struct lst_node* a, int new_data){
	if(a==NULL){
		a = malloc(sizeof(lst_node));
		a->data = new_data;
		a->next = NULL;

	}
	lst_insert(a->next, new_data);
}

void update_node(struct bst_node *a, int new_index, char word[21], int line_no){
	struct bst_node *b = NULL;
	b = search_node(a, word);
	b->index = new_index;
	lst_insert(b->list, line_no);
}

struct bst_node* search_node(struct bst_node *a, char word[]){
	if(strcasecmp(a->word, word) == 0){
		return a;
	}else if(strcasecmp(a->word, word) < 0){
		search_node(a->left, word);
	}else if(strcasecmp(a->word, word) > 0){
		search_node(a->right, word);
	}else if(a->word = NULL){
		return NULL;
	}
}*/
struct bst_node* reader(FILE *file){
	if(!file){
		fprintf(stderr, "Couldn't open file\n");
		return NULL;
	}

	struct bst_node *Ultimate_tree = NULL;

	fseek(file, 0, SEEK_END);
	int size = ftell(file);
	fseek(file, 0, SEEK_SET);

	int line_no = 1;
	int i = 0;
	char chr;
	chr = getc(file);

	char word[21];

	while(chr != EOF){
		if(chr = '\n'){
            bst_add(Ultimate_tree, NULL, word, line_no);
            memset(word, 0, 21);
            line_no = line_no+1;
		}

		if(chr == EndOfWord){
			bst_add(Ultimate_tree,  NULL, word, line_no);
			/*
			if(search_node(Ultimate_tree, word)){
				update_node(Ultimate_tree, word, line_no);
			}else{
				bst_add(Ultimate_tree, NULL, word, NULL);
			}*/

			i = 0;
		}else{
			chr = word[i];
			i = i+1;
		}
	}

	return Ultimate_tree;
}

