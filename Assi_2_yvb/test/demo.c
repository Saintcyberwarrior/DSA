#include <stdio.h>
#include <stdlib.h>

int main(){
	char name[80];

	printf("ENter your name:");
	void *a=malloc(80000000000000000);
	if(a==NULL)
		printf("CYUKA BLYAT\n");
	scanf("%s",name);
	printf("Hello %s\n",name);
	free(a);
	return 0;
	}
