function P = step(P)
    P(find(P>0))=1;
    P(find(P<=0))=0;
end

