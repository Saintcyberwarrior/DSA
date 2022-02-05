#include <stdio.h>
#include <stdlib.h>
struct Node {
	int data;
	struct Node* next;
};

void push(struct Node** a, int new_data);
int getCount(struct Node* head);
int GetNth(struct Node* a, int i);
int max(int x, int y);
int min(int x, int y);
void frre(struct Node* begin);
struct Node* add(struct Node* a, struct Node* b);

int main(){
	struct Node* p = NULL;
	push(&p, 555);
	push(&p, 465);
	push(&p, 564);
	push(&p, 597);
	struct Node* q = NULL;
	push(&q, 295);
	push(&q, 565);
	push(&q, 581);
	struct Node *r = NULL;
	printf("%d\n", getCount(p));
	printf("%d\n", getCount(p));
	printf("%d\n", getCount(q));
	r = add(p, q);

	printf("%d\n", getCount(r));

	printf("%d\n", GetNth(r, 1));
	printf("%d\n", GetNth(r, 2));
	printf("%d\n", GetNth(r, 3));
	printf("%d\n", GetNth(r, 4));


	frre(p);
	frre(q);
	frre(r);
	printf("\nSuccess!!\n");
	return 0;
}

void push(struct Node** a, int new_data){
        struct Node* new_node = (struct Node*)malloc(sizeof(struct Node));
        new_node->data = new_data;
        new_node->next = (*a);
        (*a) = new_node;
}



int getCount(struct Node* head){
	int count = 1;  // Initialize count
	struct Node* current = head;  // Initialize current
	while (current != NULL){
		if(current->next == NULL)
			return count;
		current = current->next;
		count++;
	}
	return count;
}

int GetNth(struct Node* a, int i){
         struct Node* now;
         int count = 1;
         now = a;
         for(;now != NULL; count++){
                 if(count == i){
                 return now->data;
                 }
                 now = now->next;
	 }
         fprintf(stderr,"Over the limit\n");
         return 1;
 }
int max(int x, int y){
	if (x >=y)
		return x;
	else
		return y;
}

int min(int x, int y){
	if (x<=y)
		return x;
	else
		return y;
}

void frre(struct Node* begin){
         struct Node* now= begin ;
         while(now != NULL && begin->next !=NULL){
                 now = begin->next;
                 free(begin);
                 begin=now;
         }
 }


struct Node* add(struct Node* a, struct Node* b){
	int len_a = getCount(a);
	int len_b = getCount(b);
	struct Node* c = NULL;

	int car = 0;

	for(int i = 0; i < max(len_a, len_b); i++){
		if(i < min(len_a, len_b)){
			push(&c, (GetNth(a, len_a-i) + GetNth(b, len_b-i) + car)%1000);
			car = (GetNth(a, len_a-i) + GetNth(b, len_b-i) + car)/1000;
		}else{
			if(len_a < i && i < len_b){
				push(&c, (GetNth(b, len_b-i) + car)%1000);
				car = (GetNth(b, len_b-i) + car)/1000;
			}else{
				push(&c, (GetNth(a, len_a-i) + car)%1000);
				car = (GetNth(a, len_a-i) + car)/1000;
			}
		}
	}
	if(car)
		push(&c, car);
	return c;
}

