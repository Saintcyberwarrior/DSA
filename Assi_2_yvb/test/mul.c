struct Node* mul(struct Node* a, struct Node* b){
	int len_a = getCount(a);
	int len_b = getCount(b);
//	for (int h =0; h<min(len_a, len_b); h++){
//	struct Node* t_h = NULL;
//	}
	struct Node * t = NULL;
	
	if(len_a<len_b){
		for (int i =0; i < len_a; i++){
			int car =0;
			
			for(int j= 0; j<len_b; j++){
				fpush(&t_i, ((GetNth(a, len_a-i)*GetNth(b, len_b-j))+car)%1000);
				car = ((GetNth(a, len_a-i) * GetNth(b, len_b-j)) + car)/1000;
			}
			for (int count=0; count<i; count++){
				bpush(&t_count, 000);
			}
			
			
			if(car)
				fpush(&t_i, car);
		}
		
		for(int p=1; p< len_a; p++){
			t = sum(t_0, t_p);
			t_0 = t;
			frre(t_p);
			frre(t);
		}

		return t_0;
	}
	else{
		for (int i =0; i < len_b; i++){
			int car =0;
			
			for(int j= 0; j<len_a; j++){
				fpush(&t_i, ((GetNth(b, len_b-i)*GetNth(a, len_a-j))+car)%1000);
				car = ((GetNth(b, len_b-i) * GetNth(a, len_a-j)) + car)/1000;
			}
			for (int count=0; count<i; count++){
				bpush(&t_count, 000);
			}
			
			
			if(car)
				fpush(&t_i, car);
		}
		
		for(int p=1; p< len_b; p++){
			t = sum(t_0, t_p);
			t_0 = t;
			frre(t_p);
			frre(t);
		}

		return t_0;

	}
}
