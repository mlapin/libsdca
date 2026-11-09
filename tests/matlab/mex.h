// Minimal stand-in for MATLAB's mex.h: declarations only, for compile checks.
#pragma once
#include <cstddef>
#include <cstdint>
typedef size_t mwSize;
typedef size_t mwIndex;
typedef struct mxArray_tag mxArray;
typedef enum { mxUNKNOWN_CLASS = 0, mxCELL_CLASS, mxSTRUCT_CLASS, mxLOGICAL_CLASS,
  mxCHAR_CLASS, mxVOID_CLASS, mxDOUBLE_CLASS, mxSINGLE_CLASS, mxINT8_CLASS,
  mxUINT8_CLASS, mxINT16_CLASS, mxUINT16_CLASS, mxINT32_CLASS, mxUINT32_CLASS,
  mxINT64_CLASS, mxUINT64_CLASS, mxFUNCTION_CLASS } mxClassID;
typedef enum { mxREAL = 0, mxCOMPLEX } mxComplexity;
typedef bool mxLogical;
extern "C" {
void mexErrMsgIdAndTxt(const char*, const char*, ...);
void mexWarnMsgIdAndTxt(const char*, const char*, ...);
void mexErrMsgTxt(const char*);
int mexPrintf(const char*, ...);
int mexEvalString(const char*);
void* mxCalloc(mwSize, mwSize);
void* mxMalloc(mwSize);
void mxFree(void*);
mxArray* mxCreateDoubleMatrix(mwSize, mwSize, mxComplexity);
mxArray* mxCreateNumericMatrix(mwSize, mwSize, mxClassID, mxComplexity);
mxArray* mxCreateDoubleScalar(double);
mxArray* mxCreateString(const char*);
mxArray* mxCreateStructMatrix(mwSize, mwSize, int, const char**);
mxArray* mxCreateCellMatrix(mwSize, mwSize);
mxArray* mxCreateLogicalScalar(mxLogical);
mxArray* mxDuplicateArray(const mxArray*);
void mxDestroyArray(mxArray*);
mxArray* mxGetCell(const mxArray*, mwIndex);
void mxSetCell(mxArray*, mwIndex, mxArray*);
mxClassID mxGetClassID(const mxArray*);
const char* mxGetClassName(const mxArray*);
void* mxGetData(const mxArray*);
double* mxGetPr(const mxArray*);
void mxSetData(mxArray*, void*);
void mxSetPr(mxArray*, double*);
mxArray* mxGetField(const mxArray*, mwIndex, const char*);
mxArray* mxGetFieldByNumber(const mxArray*, mwIndex, int);
int mxGetFieldNumber(const mxArray*, const char*);
int mxAddField(mxArray*, const char*);
void mxRemoveField(mxArray*, int);
const char* mxGetFieldNameByNumber(const mxArray*, int);
int mxGetNumberOfFields(const mxArray*);
void mxSetField(mxArray*, mwIndex, const char*, mxArray*);
void mxSetFieldByNumber(mxArray*, mwIndex, int, mxArray*);
mwIndex* mxGetIr(const mxArray*);
mwIndex* mxGetJc(const mxArray*);
mwSize mxGetNzmax(const mxArray*);
mwSize mxGetM(const mxArray*);
mwSize mxGetN(const mxArray*);
mwSize mxGetNumberOfElements(const mxArray*);
mwSize mxGetNumberOfDimensions(const mxArray*);
const mwSize* mxGetDimensions(const mxArray*);
void mxSetM(mxArray*, mwSize);
void mxSetN(mxArray*, mwSize);
double mxGetScalar(const mxArray*);
int mxGetString(const mxArray*, char*, mwSize);
char* mxArrayToString(const mxArray*);
bool mxIsCell(const mxArray*);
bool mxIsChar(const mxArray*);
bool mxIsDouble(const mxArray*);
bool mxIsEmpty(const mxArray*);
bool mxIsNumeric(const mxArray*);
bool mxIsSingle(const mxArray*);
bool mxIsSparse(const mxArray*);
bool mxIsStruct(const mxArray*);
bool mxIsComplex(const mxArray*);
bool mxIsLogical(const mxArray*);
bool mxIsScalar(const mxArray*);
}
