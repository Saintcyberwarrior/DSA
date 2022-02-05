#include <stdio.h>
#include <stdlib.h>

#define gh_kha_lo -1

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
/*
int len(struct Node** a){
	struct Node * b=(*a);
	int count = 1;
	while((b->next)!=NULL){
		b = b->next;
		count ++;
	}
	return count;
}
*/
int len(struct Node* a){
	int count =1;

	while((a->next!=NULL)){
		a = a->next;
		count++;
	}

	return count;
}


int max(a, b){
	if(a>b)
		return a;
		return b;
}

int min(a,b){
	if(a<b)
		return a;
		return b;
}

int geti(struct Node** b, int i){
	struct Node * a = (*b);
	int inf= 9999;
	int count = 1;
	if(i>len(a)) return gh_kha_lo;
	for(; i != count; count++){
		a = a->next;
	}
	return a->data;
}


int add(struct Node* a, struct Node* b){
	int count = 0;
	int len_a = len(a);
	int len_b = len(b);
	
	int car = 0;
	if (count < min(len_a, len_b)){
	
	}
	return 
}

int main(void){
	struct Node* p = NULL;
	push(&p, 5);
	push(&p, 4);
	push(&p, 3);
	push(&p, 3);
	push(&p, 3);
	push(&p, 2);

//	printf("%d\n", len(&p));
//	printf("%d\n", len2(p));
//	printf("%d\n", len(&p));
	printf("%d\n", geti(&p, 1));
	printf("%d\n", geti(&p, 2));
	printf("%d\n", geti(&p, 3));
	printf("%d\n", geti(&p, 4));
	printf("%d\n", geti(&p, 5));
	printf("%d\n", geti(&p, 6));
	return 0;

}


