#ifndef SPECSWD_VOIGT_H_
#define SPECSWD_VOIGT_H_

namespace specswd
{

const int _VOIGT_LOOKUP[3][3] = {
    {0, 5, 4},
    {5, 1, 3},
    {4, 3, 2}
};

/**
 * @brief mapping from index (i,j) to voigt index
 * 
 * @param i,j (i,j pair), in [0,3) 
 * @return int voigt index in [0,6)
 */
int inline voigt2(int i,int j)
{
    return _VOIGT_LOOKUP[i][j];
}

/**
 * @brief mapping c[i,j,p,q] to c21[idx]
 * 
 * @param i,j,p,q cijpq index, [0,2]
 * @return int loc in c21
 */
int inline voigt4(int i,int j, int p, int q)
{
    int m0 = _VOIGT_LOOKUP[i][j];
    int n0 = _VOIGT_LOOKUP[p][q];

    int m = m0, n = n0;
    if(m0 > n0) {
        m = n0;
        n = m0;
    }
    int idx = m * 6 + n - (m * (m + 1)) / 2;

    return idx;
}

    
} // namespace specswd


#endif