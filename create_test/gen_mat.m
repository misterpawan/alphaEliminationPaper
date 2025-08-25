for k=1:100
	n = 50;
	sparsity = 0.3;
	S = sprand(n, n, sparsity);
	S = S + eye(n,n);
	P = colamd(S);
	filename = ['test_set_colmd_' num2str(n) '_' num2str(n) '_' num2str(sparsity) '/' num2str(k) '.mat'];
	save(filename, 'S', 'P', '-v7');
end
