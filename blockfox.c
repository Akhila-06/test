/*
   block_fox.c - Block matrix multiplication using Fox's algorithm
   Compile as: mpicc -Wall -lm -o block_fox block_fox.c
   Run as:     mpiexec -n nprocs ./block_fox
   Program will prompt user to enter a filename containing the following:
   < matrix dimension n >
   < elements of matrix A >
   < elements of matrix B >
   Matrix Product C = A x B will be printed to the standard output
*/


#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define TAG 1

int main( int argc, char *argv[] ) {

  int nprocs, p, i, j, k, m, n, step;
  int rank, bcast_root, shift_source, shift_dest;
  int dimensions[2], periodic[2], coords[2];
  char filename[FILENAME_MAX];
  MPI_Comm grid_2D, row_comm;
  MPI_Datatype blocktype;
  MPI_Status status;
  FILE *file;

  MPI_Init( &argc, &argv );
  MPI_Comm_size( MPI_COMM_WORLD, &nprocs );

   /* Create pxp Cartesian grid, where p = sqrt(nprocs) */
  p = (int) sqrt( nprocs );
  dimensions[0] = dimensions[1] = p;
  periodic[0] = 1;	/* periodic along dimension 0 */
  periodic[1] = 0;
  MPI_Cart_create( MPI_COMM_WORLD, 2, dimensions, periodic, 0, &grid_2D );

  if ( grid_2D != MPI_COMM_NULL ) {

    /* Get Cartesian rank and coordinates */
    MPI_Comm_rank( grid_2D, &rank );
    MPI_Cart_coords( grid_2D, rank, 2, coords );

    if ( rank == 0 ) {
      printf("Enter filename: ");
      gets( filename );
      file = fopen( filename, "r" );
      fscanf( file, "%d", &n );
      printf( "\nMatrix size = %d x %d\n", n, n );
      printf( "Grid size = %d x %d\n", p, p );
    }

     /* Broadcast n to all processes in the grid */
     MPI_Bcast( &n, 1, MPI_INT, 0, grid_2D );

    if ( n%p != 0 ) {
      if ( rank == 0 ) {
        printf( "Grid size must divide matrix size.\n" );
        fclose( file );
      }
      MPI_Finalize( );
      exit( 1 );
    }

    /* Each local submatrix is of size m x m, where m = n/p */
    m = n/p;

    /* Allocate I/O bufferreading/printing data */
    float buf[n][n];

    /* Create vector derived datatype for moving mxm blocks between
       process 0 and other processes: blocklength = m, stride = n */
    MPI_Type_vector( m, m, n, MPI_FLOAT, &blocktype );
    MPI_Type_commit( &blocktype );

    /* Allocate space for local submatrices A, B, C, and Atemp */
    float local_A[m][m];
    float local_B[m][m];
    float local_C[m][m];
    float local_Atemp[m][m];   	/* used in broadcast step */

    /* Read, echo, and distribute matrix A */
    if ( rank == 0 ) {
      printf( "\nMatrix A:\n" );
      for ( i=0; i<n; i++ ) {
	    for ( j=0; j<n; j++) {
          fscanf( file, "%f", &buf[i][j] );
          printf( "%7.1f", buf[i][j] );
	    }
        printf( "\n" );
      }
      for ( i=0; i<p; i++ ) {
        for (j=0; j<p; j++ ) {
          /* Block (i,j) - starts at buf[i*m][j*m] */
          /*             - destined for process i*p+j */
          if ((i==0) && (j==0)) continue;
          MPI_Send( &buf[i*m][j*m], 1, blocktype, i*p+j, TAG, grid_2D );
        }
      }
    }
    if (rank > 0)
      MPI_Recv( local_A, m*m, MPI_FLOAT, 0, TAG, grid_2D, &status );
    else {
      for (i=0; i<m; i++)
        for (j=0; j<m; j++)
          local_A[i][j] = buf[i][j];
    }

    /* Read, echo, and distribute matrix B */
    if ( rank == 0 ) {
      printf( "\nMatrix B:\n" );
      for ( i=0; i<n; i++ ) {
	    for ( j=0; j<n; j++) {
	      fscanf( file, "%f", &buf[i][j] );
	      printf( "%7.1f", buf[i][j] );
	  	}
	    printf( "\n" );
      }
      fclose( file );
      for ( i=0; i<p; i++ ) {
        for (j=0; j<p; j++ ) {
          /* Block (i,j) - starts at buf[i*m][j*m] */
          /*             - destined for process i*p+j */
          if ((i==0) && (j==0)) continue;
          MPI_Send( &buf[i*m][j*m], 1, blocktype, i*p+j, TAG, grid_2D );
        }
      }
    }
    if (rank > 0)
      MPI_Recv( local_B, m*m, MPI_FLOAT, 0, TAG, grid_2D, &status );
    else {
	  for (i=0; i<m; i++)
	    for (j=0; j<m; j++)
	      local_B[i][j] = buf[i][j];
    }

    /* Initialize local submatrix C to zeros */
    for ( i=0; i<m; i++ )
      for ( j=0; j<m; j++)
        local_C[i][j] = 0.0;

    /* Create row communicators for broadcast */
    MPI_Comm_split( grid_2D, coords[0], coords[1], &row_comm );

    /* Determine source and destination of circular-shift along columns */
    MPI_Cart_shift( grid_2D, 0, -1, &shift_source, &shift_dest );

    /* Main loop */
    for ( step=0; step<p; step++ ) {

      /* Broadcast local_A submatrix of designated root to processes in same row */
      if (coords[1] == step) {
    memcpy(temp_A, local_A, m * m * sizeof(float));
}
MPI_Bcast(temp_A, m * m, MPI_FLOAT, step, row_comm);



      /* Perform local matrix multiplication */
     

for (i = 0; i < m; i++) {
    for (j = 0; j < m; j++) {
        for (k = 0; k < m; k++) {
            local_C[i * m + j] += temp_A[i * m + k] * local_B[k * m + j];
        }
    }
}

      /* Roll local submatrices B along columns */
     
MPI_Sendrecv_replace(local_B, m * m, MPI_FLOAT,
                     shift_dest, 0,
                     shift_source, 0,
                     grid_2D, &status);

    }

    /* Print product matrix C */
    MPI_Gather( local_C, m*m, MPI_FLOAT, buf, m*m, MPI_FLOAT, 0, grid_2D );
    if ( rank == 0 ) {
      printf( "\nProduct Matrix C:\n" );
      for ( i=0; i<n; i++ ) {
	    for ( j=0; j<n; j++) {
	  	  printf( "%7.1f", buf[i][j] );
	  	}
	  	printf( "\n" );
      }
    }
    MPI_Comm_free( &row_comm );
    MPI_Comm_free( &grid_2D );
    MPI_Type_free( &blocktype );
  }
  MPI_Finalize( );
  return 0;
}
