#include <stdio.h>
#include <stdlib.h>

struct Node{
	int data;
	struct Node* next;
};

void push(struct Node** a, int new_data){
	struct Node* new_node;
	new_node = (struct Node*)malloc(sizeof(struct Node));
	new_node->data = new_data;
	new_node->next = *a;
	*a = new_node;
}

int len(struct Node** a){
	int count = 0;
	while((*a)!=NULL){
	(*a) = (*a)->next;
	count ++;
	}
	return count;
}

int geti(struct Node* a, int i){
	int count = 1;
	for(; i == count; count++){
		(a) = (*a).next;
	}
	return (*a).data;
}

int main(void){
	struct Node* p = NULL;
	push(&p, 5);
	push(&p, 4);
	push(&p, 3);
	push(&p, 3);
	push(&p, 3);
	push(&p, 2);

	printf("%d\n", len(&p));
	printf("%d\n", geti(p, 1));
	printf("%d\n", geti(p, 2));
	printf("%d\n", geti(p, 3));
	printf("%d\n", geti(p, 4));
	printf("%d\n", geti(p, 5));
	printf("%d\n", geti(p, 6));
	return 0;

}


