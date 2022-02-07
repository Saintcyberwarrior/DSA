struct Node* mul(struct Node* a, struct Node *b){
	int len_a = getCount(a);
	int len_b = getCount(b);

	struct Node* t_0 = NULL;
	
	if(len_a<len_b){
		int car = 0;
		
		for(int j=0; j<len_b;j++){
			fpush(&t_0,((GetNth(a, len_a)*GetNth(b, len_b-j))+car)%1000);
			car = ((GetNth(a, len_a) * GetNth(b, len_b-j)) + car)/1000;
		}
		if(car)
			fpush(&t_0, car);
	
		for (int i =1; i < len_a; i++){
                        int car =0;
 
                        for(int j= 0; j<len_b; j++){
                                fpush(&t_1, ((GetNth(a, len_a-i)*GetNth(b, len_b-j))+car)%1000);
                                car = ((GetNth(a, len_a-i) * GetNth(b, len_b-j)) + car)/1000;
                        }
                        for (int count=1; count<i; count++){
                                bpush(&t_1, 000);
                        }
 
  
                        if(car)
                                fpush(&t_i, car);
                 
		 	t_0 = sum(t_0, t_1);
		 }

		 frre(t_1);
		 
		return t_0;
	}
	else{
		int car = 0;
		
		for(int j=0; j<len_a;j++){
			fpush(&t_0,((GetNth(b, len_b)*GetNth(a, len_a-j))+car)%1000);
			car = ((GetNth(b, len_b) * GetNth(a, len_a-j)) + car)/1000;
		}
		if(car)
			fpush(&t_0, car);
	
		for (int i =1; i < len_b; i++){
                        int car =0;
 
                        for(int j= 0; j<len_a; j++){
                                fpush(&t_1, ((GetNth(b, len_b-i)*GetNth(a, len_a-j))+car)%1000);
                                car = ((GetNth(b, len_b-i) * GetNth(a, len_a-j)) + car)/1000;
                        }
                        for (int count=1; count<i; count++){
                                bpush(&t_1, 000);
                        }
 
  
                        if(car)
                                fpush(&t_i, car);
                 
		 	t_0 = sum(t_0, t_1);
		 }

		 frre(t_1);
		 
		return t_0;
	}






}
