#include <stdio.h>
#include <stdlib.h>
#include <float.h> //for DBL_MAX
#include <omp.h>
#include <string.h>

struct p {
    double x, y; // coordinates
    int cluster; // cluster of the point
    double minDist; // distance from the cluster
};

typedef struct p Point;

Point Point_new(double x, double y) // constructor with arguments
{
    Point p;
    p.x = x;
    p.y = y;
    p.cluster = -1;
    p.minDist = DBL_MAX;

    return p;

}

double distance(Point p1, Point p2) {
        return (p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y);
}

typedef struct 
{
    Point* data;
    unsigned long long size;

} VectorPoint;


VectorPoint* VectorPoint_new(unsigned long long size)
{
    VectorPoint* v = malloc(sizeof(VectorPoint));

    if (v)
    {
        v->data = malloc(size*sizeof(Point));
        v->size = size;
    }

    return v;
}

void VectorPoint_delete(VectorPoint* v)
{
    if(v)
    {
        free(v->data);
        free(v);
    }
}

unsigned long long VectorPoint_resize(VectorPoint* v, unsigned long long size) 
{  
    if(v)
    {
        Point* p = realloc(v->data, (size+1)*sizeof(Point));
        if(p)
        {
            v->data = p;
            v->size = size+1;

        }

        return v->size;
    }

    return 0;
}

Point VectorPoint_get(VectorPoint* v, unsigned long long n)
{
    if (v && n < v->size)
    {
        return v->data[n];
    }
}

void VectorPoint_set(VectorPoint* v, unsigned long long n, Point x)
{
    if (v)
    {
        if(n > (v->size)-1)
        {
            VectorPoint_resize(v, n);

        }
        v->data[n] = x;
    }
}

const char* get_token(char* line, int n)
{
    const char* tok;
    for (tok=strtok(line, ","); tok && *tok; tok = strtok(NULL,",\n"))
    {
        if(!--n)
            return tok;
    }
    return NULL;
}

VectorPoint* read_csv(char * filename)
{
    char str[1024];
    FILE* f = fopen(filename, "r");
    Point p;
    int pos = 0;
    VectorPoint* v = VectorPoint_new(1);
        

    while (fgets(str, 1024, f))
    {
        char* tmp = strdup(str);
        p = Point_new(strtod(get_token(tmp, 1), NULL), strtod(get_token(tmp,2), NULL));

        VectorPoint_set(v, pos,p);
        pos +=1;

        free(tmp);

    }
 
    return v;  
}


VectorPoint* KMeans(VectorPoint*v, int epochs, int k, int num_thr)
{  
    VectorPoint* centroids = VectorPoint_new(k);
    for(int c=0; c<k; c++) //initializing centroids
    {
        VectorPoint_set(centroids,c, VectorPoint_get(v, rand()%v->size));
    }

    for (int i=0; i<epochs; i++)
    {       
            for(int c=0; c<centroids->size; c++)
            {   
                Point centroid = VectorPoint_get(centroids,c);

                #pragma omp parallel for num_threads(num_thr) schedule(static, (int)v->size/num_thr) 
                for(int j=0; j<v->size; j++)
                {   
                    Point point = VectorPoint_get(v,j);
                    double dist = distance(centroid, point);

                    if(dist<point.minDist)
                    {   
                        point.minDist = dist;
                        point.cluster = c;
                        VectorPoint_set(v,j,point);

                    }
                    
                
                }

            }    

        //second part
        
        int* clusters_points = malloc(sizeof(int)*k);
        double* sumX = malloc(sizeof(double)*k);
        double* sumY = malloc(sizeof(double)*k);

        for(int m=0; m<k; m++)
        {
            clusters_points[m]=0;
            sumX[m]=0;
            sumY[m]=0;
        }

        
        #pragma omp parallel for num_threads(num_thr) schedule(static, (int)v->size/num_thr) 
        for(int j=0; j<v->size; j++)
        {
            Point point = VectorPoint_get(v,j);

            #pragma omp critical
            {
                clusters_points[point.cluster] +=1; 
                sumX[point.cluster] += point.x; 
                sumY[point.cluster] += point.y;
                point.minDist = DBL_MAX;
                VectorPoint_set(v,j,point); 
            }
            
        }

        for(int c=0; c<centroids->size; c++)
        {
            Point centroid = VectorPoint_get(v,c);
            centroid.x = sumX[c]/clusters_points[c];
            centroid.y = sumY[c]/clusters_points[c];

            VectorPoint_set(centroids,c,centroid);

        }
        
    }

    return v;
}

void write_on_file(VectorPoint *v)
{
    FILE *fpt;
    fpt = fopen("k-means2_output.csv", "w+");

    fprintf(fpt, "x,y,Cluster\n");

    for(int j=0; j<v->size; j++)
    {
        Point p = VectorPoint_get(v,j);
        fprintf(fpt,"%f, %f, %d\n", p.x, p.y, p.cluster);

    }

    fclose(fpt);
}

int main() 
{   
    srand(42);

    int * dataset_sizes = malloc(sizeof(int)*4);
    dataset_sizes[0] = 5000;
    dataset_sizes[1] = 15000;
    dataset_sizes[2] = 100000;
    dataset_sizes[3] = 500000;
    
    char result[100];
    char number[10];
    char nolabel[] = "_nolabel.csv";

    int k=3;
    int epochs = 10;

    double start, stop;
    printf("\n");
    printf("%s\n\n", "Second solution for parallel K-Means");
    printf("%-50s %s\n", "", "Number of threads");
    printf("\n");
    printf("%-1s %s\n", "","Dataset size");
    printf("%-24s %-22s %-20s %-20s %s\n", "", "N=1", "N=2", "N=3", "N=4");

    for(int i=0; i < 4; i++)
    {   
        sprintf(number,"%d",dataset_sizes[i]);
        strcpy(result,"gen_data");
        strcat(result, number);
        strcat(result, nolabel);

        printf("%-2s %-20d", "",dataset_sizes[i]);
        
        VectorPoint* v = read_csv(result);
        for(int j=0; j<4;j++)
        {
            start = omp_get_wtime();
            v=KMeans(v, epochs, k, j+1);
            stop = omp_get_wtime();

            write_on_file(v);
            printf("%-22f", (stop-start));
        }
        printf("\n");
        
    }

}