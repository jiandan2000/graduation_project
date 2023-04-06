float ndot(const float *a,const float *b,int factors){
    float sum=0;
    for(int i=0;i<factors;i++)
        sum=sum+a[i]*b[i];

    return sum;
}

__kernel void LS_CG(const int factors, int user_count, int item_count, float * X, const float * Y, 
                    float * YtY,const int * indptr, const int * indices, const float * data,int cg_steps) {

    int groupSize=get_num_groups(0);
    int groupid=get_group_id(0);
    int threadid=get_local_id(0);

    //if(groupid==0)
        //printf("groupSize:%d groupid:%d threadid:%d\n",groupSize,groupid,threadid);
        
    __local float Ap[10];
    __local float r[10];
    __local float p[10];

    //local float *Ap=(local float*) malloc(sizeof(float) * factors);
    //__local float* Ap = (__local float*) malloc(sizeof(float) * factors);
    //local float *r=(local float*) malloc(sizeof(float) * factors);
    //__local float* r = (__local float*) malloc(sizeof(float) * factors);
    //local float *p=(local float*) malloc(sizeof(float) * factors);
    //__local float* p = (__local float*) malloc(sizeof(float) * factors);

    for (int u = groupid; u < user_count; u +=groupSize ) {
        float * x = &X[u * factors];
        //printf("%d\n",u);

        float temp = 0;
        for (int i = 0; i < factors; ++i) {
            temp -= x[i] * YtY[i * factors + threadid];
        }
        for (int index = indptr[u]; index < indptr[u + 1]; ++index) {
            const float * Yi = &Y[indices[index] * factors];
            float confidence = data[index];
            temp += (confidence - (confidence - 1) * ndot(Yi, x,factors)) * Yi[threadid];
        }
        p[threadid] = r[threadid] = temp;

        float rsold = ndot(r, r,factors);

        for (int it = 0; it < cg_steps; ++it) {
            Ap[threadid] = 0;
            for (int i = 0; i < factors; ++i) {
                Ap[threadid] += p[i] * YtY[i * factors + threadid];
            }
            for (int index = indptr[u]; index < indptr[u + 1]; ++index) {
                const float * Yi = &Y[indices[index] * factors];
                Ap[threadid] += (data[index] - 1) * ndot(Yi, p, factors) * Yi[threadid];
            }

            float alpha = rsold / ndot(p, Ap, factors);
            x[threadid] += alpha * p[threadid];
            r[threadid] -= alpha * Ap[threadid];
            float rsnew = ndot(r, r, factors);
            p[threadid] = r[threadid] + (rsnew/rsold) * p[threadid];
            rsold = rsnew;
        }
    }
}