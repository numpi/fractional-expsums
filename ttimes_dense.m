function T = ttimes_dense(A, T, ind, divide)
% Multiply the tensor T in the mode ind by the matrix A
	if size(T, ind) ~= size(A, 2)
		error('TTIMES::incompatible dimensions')
    end

%     if exist('tensorprod', 'builtin')
%         if exist('divide', 'var') && divide
%             T = tensorprod(inv(A), T, 2, ind);
%         else
%             T = tensorprod(A, T, 2, ind);
%         end
% 
%         d = ndims(T);
%         p = [ind, 1:ind-1, ind+1:d];
% 	    ip(p) = 1:d;
% 	    T = permute(T, ip);
%         return;
%     end

    sz=size(T);
    sz(ind)=size(A, 1);
    if exist('divide', 'var') && divide
        T = fold(A \ unfold(T, ind), sz, ind);    
    else
        T = fold(A * unfold(T, ind), sz, ind);
    end
end
