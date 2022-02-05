#include <stdio.h>
#include <stdlib.h>
struct Node {
	int data;
	struct Node* next;
};

void fpush(struct Node** a, int new_data);
void bpush(struct Node** a, int new_data);
int getCount(struct Node* head);
int GetNth(struct Node* a, int i);
int max(int x, int y);
int min(int x, int y);
void frre(struct Node* begin);
struct Node* add(struct Node* a, struct Node* b);
struct Node* mul(struct Node* a, struct Node* b);
void print(struct Node *temp);

int main(){
	struct Node* p = NULL;
	fpush(&p, 999);
	fpush(&p, 4);
	fpush(&p, 6);
	fpush(&p, 9);
	struct Node* q = NULL;
	fpush(&q, 1);
	struct Node *s = NULL;
	s = add(q,p);

	print(p);
	print(q);
	print(s);

	frre(p);
	frre(q);
	frre(s);
	return 0;
}

void print(struct Node *temp){
	while(temp){
		if(!(temp->next)){
			printf("%d$\n",temp->data);
			return;
		}
		printf("%d,",temp->data);
		temp = temp->next;
	}
}


void fpush(struct Node** a, int new_data){
        struct Node* new_node = (struct Node*)malloc(sizeof(struct Node));
        new_node->data = new_data;
        new_node->next = (*a);
        (*a) = new_node;
}

void bpush(struct Node** a, int new_data){
        struct Node* new_node = (struct Node*)malloc(sizeof(struct Node));
        new_node->data = new_data;
        new_node->next = NULL;
	if((*a)==NULL){
		(*a) = new_node;
		return;
	}
	struct Node* temp = (*a);
	while(temp->next){
		temp = temp->next;
	}
	temp->next = new_node;
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
			fpush(&c, (GetNth(a, len_a-i) + GetNth(b, len_b-i) + car)%1000);
			car = (GetNth(a, len_a-i) + GetNth(b, len_b-i) + car)/1000;
		}else{
			if(len_b>i){
				fpush(&c, (GetNth(b, len_b-i) + car)%1000);
				car = (GetNth(b, len_b-i) + car)/1000;
			}else{
				fpush(&c, (GetNth(a, len_a-i) + car)%1000);
				car = (GetNth(a, len_a-i) + car)/1000;
			}
		}
	}
	if(car)
		fpush(&c, car);
	return c;
}

struct Node* mul(struct Node* a, struct Node* b){
	int len_a = getCount(a);
	int len_b = getCount(b);
	struct Node* c = NULL;

	int car = 0;

	for(int i = 0; i < max(len_a, len_b); i++){
		if(i < min(len_a, len_b)){
			fpush(&c, (GetNth(a, len_a-i) * GetNth(b, len_b-i) + car)%1000);
			car = (GetNth(a, len_a-i) * GetNth(b, len_b-i) + car)/1000;
		}else{
			if(len_b>=i){
				fpush(&c, (GetNth(b, len_b-i) + car)%1000);
				car = (GetNth(b, len_b-i) + car)/1000;
			}else{
				fpush(&c, (GetNth(a, len_a-i) + car)%1000);
				car = (GetNth(a, len_a-i) + car)/1000;
			}
		}
	}
	if(car)
		fpush(&c, car);
	return c;
}

