#include <iostream>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/opencv.hpp>


const cv::Scalar colorWaterBlue(230, 255, 102);
const cv::Scalar colorBlue(255, 0, 0);
const cv::Scalar colorYellow(0, 255, 255);
const cv::Scalar colorRed(0, 0, 255);
const cv::Scalar colorBlack(0, 0, 0);
const cv::Scalar colorGray(200, 200, 200);
const cv::Scalar colorSystem(240, 240, 240);
const cv::Scalar colorGreen(0, 255, 0);
const cv::Scalar colorWhite(255, 255, 255);
const cv::Scalar colorPurple(214, 112, 218);
const cv::Scalar colorGoldenrod(15, 185, 255);

#define VISION_TOLERANCE 0.0000001
#define D2R (CV_PI / 180.0)
#define R2D (180.0 / CV_PI)
#define MATCH_CANDIDATE_NUM 5

struct s_TemplData
{
	std::vector<cv::Mat> vecPyramid;
	std::vector<cv::Scalar> vecTemplMean;
	std::vector<double> vecTemplNorm;
	std::vector<double> vecInvArea;
	std::vector<bool> vecResultEqual1;
	bool bIsPatternLearned = false;
	int iBorderColor;
	void clear()
	{
		std::vector<cv::Mat>().swap(vecPyramid);
		std::vector<double>().swap(vecTemplNorm);
		std::vector<double>().swap(vecInvArea);
		std::vector<cv::Scalar>().swap(vecTemplMean);
		std::vector<bool>().swap(vecResultEqual1);
	}
	void resize(int iSize)
	{
		vecTemplMean.resize(iSize);
		vecTemplNorm.resize(iSize, 0);
		vecInvArea.resize(iSize, 1);
		vecResultEqual1.resize(iSize, false);
	}

	s_TemplData()
	{
		bIsPatternLearned = false;
	}
};
struct s_MatchParameter
{
	cv::Point2d pt;
	double dMatchScore;
	double dMatchAngle;
	//Mat matRotatedSrc;
	cv::Rect rectRoi;
	double dAngleStart;
	double dAngleEnd;
	cv::RotatedRect rectR;
	cv::Rect rectBounding;
	bool bDelete;

	double vecResult[3][3];//for subpixel
	int iMaxScoreIndex;//for subpixel
	bool bPosOnBorder;
	cv::Point2d ptSubPixel;
	double dNewAngle;

	s_MatchParameter(cv::Point2f ptMinMax, double dScore, double dAngle)//, Mat matRotatedSrc = Mat ())
	{
		pt = ptMinMax;
		dMatchScore = dScore;
		dMatchAngle = dAngle;

		bDelete = false;
		dNewAngle = 0.0;

		bPosOnBorder = false;
	}
	s_MatchParameter()
	{
		double dMatchScore = 0;
		double dMatchAngle = 0;
	}
	~s_MatchParameter()
	{

	}
};
struct s_SingleTargetMatch
{
	cv::Point2d ptLT, ptRT, ptRB, ptLB, ptCenter;
	double dMatchedAngle;
	double dMatchScore;
};
struct s_BlockMax
{
	struct Block
	{
		cv::Rect rect;
		double dMax;
		cv::Point ptMaxLoc;
		Block()
		{}
		Block(cv::Rect rect_, double dMax_, cv::Point ptMaxLoc_)
		{
			rect = rect_;
			dMax = dMax_;
			ptMaxLoc = ptMaxLoc_;
		}
	};
	s_BlockMax()
	{}
	std::vector<Block> vecBlock;
	cv::Mat matSrc;
	s_BlockMax(cv::Mat matSrc_, cv::Size sizeTemplate)
	{
		matSrc = matSrc_;
		//?matSrc ????block????????
		int iBlockW = sizeTemplate.width * 2;
		int iBlockH = sizeTemplate.height * 2;

		int iCol = matSrc.cols / iBlockW;
		bool bHResidue = matSrc.cols % iBlockW != 0;

		int iRow = matSrc.rows / iBlockH;
		bool bVResidue = matSrc.rows % iBlockH != 0;

		if (iCol == 0 || iRow == 0)
		{
			vecBlock.clear();
			return;
		}

		vecBlock.resize(iCol * iRow);
		int iCount = 0;
		for (int y = 0; y < iRow; y++)
		{
			for (int x = 0; x < iCol; x++)
			{
				cv::Rect rectBlock(x * iBlockW, y * iBlockH, iBlockW, iBlockH);
				vecBlock[iCount].rect = rectBlock;
				minMaxLoc(matSrc(rectBlock), 0, &vecBlock[iCount].dMax, 0, &vecBlock[iCount].ptMaxLoc);
				vecBlock[iCount].ptMaxLoc += rectBlock.tl();
				iCount++;
			}
		}
		if (bHResidue && bVResidue)
		{
			cv::Rect rectRight(iCol * iBlockW, 0, matSrc.cols - iCol * iBlockW, matSrc.rows);
			Block blockRight;
			blockRight.rect = rectRight;
			minMaxLoc(matSrc(rectRight), 0, &blockRight.dMax, 0, &blockRight.ptMaxLoc);
			blockRight.ptMaxLoc += rectRight.tl();
			vecBlock.push_back(blockRight);

			cv::Rect rectBottom(0, iRow * iBlockH, iCol * iBlockW, matSrc.rows - iRow * iBlockH);
			Block blockBottom;
			blockBottom.rect = rectBottom;
			minMaxLoc(matSrc(rectBottom), 0, &blockBottom.dMax, 0, &blockBottom.ptMaxLoc);
			blockBottom.ptMaxLoc += rectBottom.tl();
			vecBlock.push_back(blockBottom);
		}
		else if (bHResidue)
		{
			cv::Rect rectRight(iCol * iBlockW, 0, matSrc.cols - iCol * iBlockW, matSrc.rows);
			Block blockRight;
			blockRight.rect = rectRight;
			minMaxLoc(matSrc(rectRight), 0, &blockRight.dMax, 0, &blockRight.ptMaxLoc);
			blockRight.ptMaxLoc += rectRight.tl();
			vecBlock.push_back(blockRight);
		}
		else
		{
			cv::Rect rectBottom(0, iRow * iBlockH, matSrc.cols, matSrc.rows - iRow * iBlockH);
			Block blockBottom;
			blockBottom.rect = rectBottom;
			minMaxLoc(matSrc(rectBottom), 0, &blockBottom.dMax, 0, &blockBottom.ptMaxLoc);
			blockBottom.ptMaxLoc += rectBottom.tl();
			vecBlock.push_back(blockBottom);
		}
	}
	void UpdateMax(cv::Rect rectIgnore)
	{
		if (vecBlock.size() == 0)
			return;
		//?????rectIgnore???block
		int iSize = vecBlock.size();
		for (int i = 0; i < iSize; i++)
		{
			cv::Rect rectIntersec = rectIgnore & vecBlock[i].rect;
			//???
			if (rectIntersec.width == 0 && rectIntersec.height == 0)
				continue;
			//?????????????
			minMaxLoc(matSrc(vecBlock[i].rect), 0, &vecBlock[i].dMax, 0, &vecBlock[i].ptMaxLoc);
			vecBlock[i].ptMaxLoc += vecBlock[i].rect.tl();
		}
	}
	void GetMaxValueLoc(double& dMax, cv::Point& ptMaxLoc)
	{
		int iSize = vecBlock.size();
		if (iSize == 0)
		{
			minMaxLoc(matSrc, 0, &dMax, 0, &ptMaxLoc);
			return;
		}
		//?block?????
		int iIndex = 0;
		dMax = vecBlock[0].dMax;
		for (int i = 1; i < iSize; i++)
		{
			if (vecBlock[i].dMax >= dMax)
			{
				iIndex = i;
				dMax = vecBlock[i].dMax;
			}
		}
		ptMaxLoc = vecBlock[iIndex].ptMaxLoc;
	}
};

int GetTopLayer(cv::Mat* matTempl, int iMinDstLength)
{
	int iTopLayer = 0;
	int iMinReduceArea = iMinDstLength * iMinDstLength;
	int iArea = matTempl->cols * matTempl->rows;
	while (iArea > iMinReduceArea)
	{
		iArea /= 4;
		iTopLayer++;
	}
	return iTopLayer;
}


cv::Point2f ptRotatePt2f(cv::Point2f ptInput, cv::Point2f ptOrg, double dAngle)
{
	double dWidth = ptOrg.x * 2;
	double dHeight = ptOrg.y * 2;
	double dY1 = dHeight - ptInput.y, dY2 = dHeight - ptOrg.y;

	double dX = (ptInput.x - ptOrg.x) * cos(dAngle) - (dY1 - ptOrg.y) * sin(dAngle) + ptOrg.x;
	double dY = (ptInput.x - ptOrg.x) * sin(dAngle) + (dY1 - ptOrg.y) * cos(dAngle) + dY2;

	dY = -dY + dHeight;
	return cv::Point2f((float)dX, (float)dY);
}


cv::Size GetBestRotationSize(cv::Size sizeSrc, cv::Size sizeDst, double dRAngle)
{
	double dRAngle_radian = dRAngle * D2R;
	cv::Point ptLT(0, 0), ptLB(0, sizeSrc.height - 1), ptRB(sizeSrc.width - 1, sizeSrc.height - 1), ptRT(sizeSrc.width - 1, 0);
	cv::Point2f ptCenter((sizeSrc.width - 1) / 2.0f, (sizeSrc.height - 1) / 2.0f);
	cv::Point2f ptLT_R = ptRotatePt2f(cv::Point2f(ptLT), ptCenter, dRAngle_radian);
	cv::Point2f ptLB_R = ptRotatePt2f(cv::Point2f(ptLB), ptCenter, dRAngle_radian);
	cv::Point2f ptRB_R = ptRotatePt2f(cv::Point2f(ptRB), ptCenter, dRAngle_radian);
	cv::Point2f ptRT_R = ptRotatePt2f(cv::Point2f(ptRT), ptCenter, dRAngle_radian);

	float fTopY = std::max(std::max(ptLT_R.y, ptLB_R.y), std::max(ptRB_R.y, ptRT_R.y));
	float fBottomY = std::min(std::min(ptLT_R.y, ptLB_R.y), std::min(ptRB_R.y, ptRT_R.y));
	float fRightX = std::max(std::max(ptLT_R.x, ptLB_R.x), std::max(ptRB_R.x, ptRT_R.x));
	float fLeftX = std::min(std::min(ptLT_R.x, ptLB_R.x), std::min(ptRB_R.x, ptRT_R.x));

	if (dRAngle > 360)
		dRAngle -= 360;
	else if (dRAngle < 0)
		dRAngle += 360;

	if (fabs(fabs(dRAngle) - 90) < VISION_TOLERANCE || fabs(fabs(dRAngle) - 270) < VISION_TOLERANCE)
	{
		return cv::Size(sizeSrc.height, sizeSrc.width);
	}
	else if (fabs(dRAngle) < VISION_TOLERANCE || fabs(fabs(dRAngle) - 180) < VISION_TOLERANCE)
	{
		return sizeSrc;
	}

	double dAngle = dRAngle;

	if (dAngle > 0 && dAngle < 90)
	{
		;
	}
	else if (dAngle > 90 && dAngle < 180)
	{
		dAngle -= 90;
	}
	else if (dAngle > 180 && dAngle < 270)
	{
		dAngle -= 180;
	}
	else if (dAngle > 270 && dAngle < 360)
	{
		dAngle -= 270;
	}
	else//Debug
	{
		std::cout << "Unknown error in GetBestRotationSize" << std::endl;
	}

	float fH1 = sizeDst.width * sin(dAngle * D2R) * cos(dAngle * D2R);
	float fH2 = sizeDst.height * sin(dAngle * D2R) * cos(dAngle * D2R);

	int iHalfHeight = (int)ceil(fTopY - ptCenter.y - fH1);
	int iHalfWidth = (int)ceil(fRightX - ptCenter.x - fH2);

	cv::Size sizeRet(iHalfWidth * 2, iHalfHeight * 2);

	bool bWrongSize = (sizeDst.width < sizeRet.width && sizeDst.height > sizeRet.height)
		|| (sizeDst.width > sizeRet.width && sizeDst.height < sizeRet.height
			|| sizeDst.area() > sizeRet.area());
	if (bWrongSize)
		sizeRet = cv::Size(int(fRightX - fLeftX + 0.5), int(fTopY - fBottomY + 0.5));

	return sizeRet;
}

inline int _mm_hsum_epi32(__m128i V)      // V3 V2 V1 V0
{
	// ??????????_mm_extract_epi32???
	__m128i T = _mm_add_epi32(V, _mm_srli_si128(V, 8));  // V3+V1   V2+V0  V1  V0  
	T = _mm_add_epi32(T, _mm_srli_si128(T, 4));    // V3+V1+V2+V0  V2+V0+V1 V1+V0 V0 
	return _mm_cvtsi128_si32(T);       // ???? 
}

inline int IM_Conv_SIMD(unsigned char* pCharKernel, unsigned char* pCharConv, int iLength)
{
	const int iBlockSize = 16, Block = iLength / iBlockSize;
	__m128i SumV = _mm_setzero_si128();
	__m128i Zero = _mm_setzero_si128();
	for (int Y = 0; Y < Block * iBlockSize; Y += iBlockSize)
	{
		__m128i SrcK = _mm_loadu_si128((__m128i*)(pCharKernel + Y));
		__m128i SrcC = _mm_loadu_si128((__m128i*)(pCharConv + Y));
		__m128i SrcK_L = _mm_unpacklo_epi8(SrcK, Zero);
		__m128i SrcK_H = _mm_unpackhi_epi8(SrcK, Zero);
		__m128i SrcC_L = _mm_unpacklo_epi8(SrcC, Zero);
		__m128i SrcC_H = _mm_unpackhi_epi8(SrcC, Zero);
		__m128i SumT = _mm_add_epi32(_mm_madd_epi16(SrcK_L, SrcC_L), _mm_madd_epi16(SrcK_H, SrcC_H));
		SumV = _mm_add_epi32(SumV, SumT);
	}
	int Sum = _mm_hsum_epi32(SumV);
	for (int Y = Block * iBlockSize; Y < iLength; Y++)
	{
		Sum += pCharKernel[Y] * pCharConv[Y];
	}
	return Sum;
}

void CCOEFF_Denominator(cv::Mat& matSrc, s_TemplData* pTemplData, cv::Mat& matResult, int iLayer)
{
	if (pTemplData->vecResultEqual1[iLayer])
	{
		matResult = cv::Scalar::all(1);
		return;
	}
	double* q0 = 0, * q1 = 0, * q2 = 0, * q3 = 0;

	cv::Mat sum, sqsum;
	integral(matSrc, sum, sqsum, CV_64F);

	q0 = (double*)sqsum.data;
	q1 = q0 + pTemplData->vecPyramid[iLayer].cols;
	q2 = (double*)(sqsum.data + pTemplData->vecPyramid[iLayer].rows * sqsum.step);
	q3 = q2 + pTemplData->vecPyramid[iLayer].cols;

	double* p0 = (double*)sum.data;
	double* p1 = p0 + pTemplData->vecPyramid[iLayer].cols;
	double* p2 = (double*)(sum.data + pTemplData->vecPyramid[iLayer].rows * sum.step);
	double* p3 = p2 + pTemplData->vecPyramid[iLayer].cols;

	int sumstep = sum.data ? (int)(sum.step / sizeof(double)) : 0;
	int sqstep = sqsum.data ? (int)(sqsum.step / sizeof(double)) : 0;

	//
	double dTemplMean0 = pTemplData->vecTemplMean[iLayer][0];
	double dTemplNorm = pTemplData->vecTemplNorm[iLayer];
	double dInvArea = pTemplData->vecInvArea[iLayer];
	//

	int i, j;
	for (i = 0; i < matResult.rows; i++)
	{
		float* rrow = matResult.ptr<float>(i);
		int idx = i * sumstep;
		int idx2 = i * sqstep;

		for (j = 0; j < matResult.cols; j += 1, idx += 1, idx2 += 1)
		{
			double num = rrow[j], t;
			double wndMean2 = 0, wndSum2 = 0;

			t = p0[idx] - p1[idx] - p2[idx] + p3[idx];
			wndMean2 += t * t;
			num -= t * dTemplMean0;
			wndMean2 *= dInvArea;


			t = q0[idx2] - q1[idx2] - q2[idx2] + q3[idx2];
			wndSum2 += t;


			//t = std::sqrt (MAX (wndSum2 - wndMean2, 0)) * dTemplNorm;

			double diff2 = MAX(wndSum2 - wndMean2, 0);
			if (diff2 <= std::min(0.5, 10 * FLT_EPSILON * wndSum2))
				t = 0; // avoid rounding errors
			else
				t = std::sqrt(diff2) * dTemplNorm;

			if (fabs(num) < t)
				num /= t;
			else if (fabs(num) < t * 1.125)
				num = num > 0 ? 1 : -1;
			else
				num = 0;

			rrow[j] = (float)num;
		}
	}
}


void MatchTemplate(cv::Mat& matSrc, s_TemplData* pTemplData, cv::Mat& matResult, int iLayer, bool bUseSIMD)
{
	// CBB 
	bool m_ckSIMD = false;
	if (m_ckSIMD && bUseSIMD)
	{
		//From ImageShop
		matResult.create(matSrc.rows - pTemplData->vecPyramid[iLayer].rows + 1,
			matSrc.cols - pTemplData->vecPyramid[iLayer].cols + 1, CV_32FC1);
		matResult.setTo(0);
		cv::Mat& matTemplate = pTemplData->vecPyramid[iLayer];

		int  t_r_end = matTemplate.rows, t_r = 0;
		for (int r = 0; r < matResult.rows; r++)
		{
			float* r_matResult = matResult.ptr<float>(r);
			uchar* r_source = matSrc.ptr<uchar>(r);
			uchar* r_template, * r_sub_source;
			for (int c = 0; c < matResult.cols; ++c, ++r_matResult, ++r_source)
			{
				r_template = matTemplate.ptr<uchar>();
				r_sub_source = r_source;
				for (t_r = 0; t_r < t_r_end; ++t_r, r_sub_source += matSrc.cols, r_template += matTemplate.cols)
				{
					*r_matResult = *r_matResult + IM_Conv_SIMD(r_template, r_sub_source, matTemplate.cols);
				}
			}
		}
		//From ImageShop
	}
	else
		matchTemplate(matSrc, pTemplData->vecPyramid[iLayer], matResult, CV_TM_CCORR);

	/*Mat diff;
	absdiff(matResult, matResult, diff);
	double dMaxValue;
	minMaxLoc(diff, 0, &dMaxValue, 0,0);*/
	CCOEFF_Denominator(matSrc, pTemplData, matResult, iLayer);
}

cv::Point GetNextMaxLoc(cv::Mat& matResult, cv::Point ptMaxLoc, cv::Size sizeTemplate, double& dMaxValue, double dMaxOverlap, s_BlockMax& blockMax)
{
	//?????????????
	int iStartX = int(ptMaxLoc.x - sizeTemplate.width * (1 - dMaxOverlap));
	int iStartY = int(ptMaxLoc.y - sizeTemplate.height * (1 - dMaxOverlap));
	cv::Rect rectIgnore(iStartX, iStartY, int(2 * sizeTemplate.width * (1 - dMaxOverlap))
		, int(2 * sizeTemplate.height * (1 - dMaxOverlap)));
	//??
	rectangle(matResult, rectIgnore, cv::Scalar(-1), CV_FILLED);
	blockMax.UpdateMax(rectIgnore);
	cv::Point ptReturn;
	blockMax.GetMaxValueLoc(dMaxValue, ptReturn);
	return ptReturn;
}

bool comparePtWithAngle(const std::pair<cv::Point2f, double> lhs, const std::pair<cv::Point2f, double> rhs) { return lhs.second < rhs.second; }


void SortPtWithCenter(std::vector<cv::Point2f>& vecSort)
{
	int iSize = (int)vecSort.size();
	cv::Point2f ptCenter;
	for (int i = 0; i < iSize; i++)
		ptCenter += vecSort[i];
	ptCenter /= iSize;

	cv::Point2f vecX(1, 0);

	std::vector<std::pair<cv::Point2f, double>> vecPtAngle(iSize);
	for (int i = 0; i < iSize; i++)
	{
		vecPtAngle[i].first = vecSort[i];//pt
		cv::Point2f vec1(vecSort[i].x - ptCenter.x, vecSort[i].y - ptCenter.y);
		float fNormVec1 = vec1.x * vec1.x + vec1.y * vec1.y;
		float fDot = vec1.x;

		if (vec1.y < 0)//????????
		{
			vecPtAngle[i].second = acos(fDot / fNormVec1) * R2D;
		}
		else if (vec1.y > 0)//??
		{
			vecPtAngle[i].second = 360 - acos(fDot / fNormVec1) * R2D;
		}
		else//???????Y
		{
			if (vec1.x - ptCenter.x > 0)
				vecPtAngle[i].second = 0;
			else
				vecPtAngle[i].second = 180;
		}

	}
	std::sort(vecPtAngle.begin(), vecPtAngle.end(), comparePtWithAngle);
	for (int i = 0; i < iSize; i++)
		vecSort[i] = vecPtAngle[i].first;
}
cv::Point GetNextMaxLoc(cv::Mat& matResult, cv::Point ptMaxLoc, cv::Size sizeTemplate, double& dMaxValue, double dMaxOverlap)
{
	//??????????? : +-??????
	//int iStartX = ptMaxLoc.x - iTemplateW;
	//int iStartY = ptMaxLoc.y - iTemplateH;
	//int iEndX = ptMaxLoc.x + iTemplateW;

	//int iEndY = ptMaxLoc.y + iTemplateH;
	////??
	//rectangle (matResult, Rect (iStartX, iStartY, 2 * iTemplateW * (1-dMaxOverlap * 2), 2 * iTemplateH * (1-dMaxOverlap * 2)), Scalar (dMinValue), CV_FILLED);
	////????????
	//Point ptNewMaxLoc;
	//minMaxLoc (matResult, 0, &dMaxValue, 0, &ptNewMaxLoc);
	//return ptNewMaxLoc;

	//?????????????
	int iStartX = ptMaxLoc.x - sizeTemplate.width * (1 - dMaxOverlap);
	int iStartY = ptMaxLoc.y - sizeTemplate.height * (1 - dMaxOverlap);
	//??
	rectangle(matResult, cv::Rect(iStartX, iStartY, 2 * sizeTemplate.width * (1 - dMaxOverlap), 2 * sizeTemplate.height * (1 - dMaxOverlap)), cv::Scalar(-1), CV_FILLED);
	//????????
	cv::Point ptNewMaxLoc;
	minMaxLoc(matResult, 0, &dMaxValue, 0, &ptNewMaxLoc);
	return ptNewMaxLoc;
}

bool compareScoreBig2Small(const s_MatchParameter& lhs, const s_MatchParameter& rhs) { return  lhs.dMatchScore > rhs.dMatchScore; }
bool compareMatchResultByPos(const s_SingleTargetMatch& lhs, const s_SingleTargetMatch& rhs)
{
	double dTol = 2;
	if (fabs(lhs.ptCenter.y - rhs.ptCenter.y) <= dTol)
		return lhs.ptCenter.x < rhs.ptCenter.x;
	else
		return lhs.ptCenter.y < rhs.ptCenter.y;

};
bool compareMatchResultByScore(const s_SingleTargetMatch& lhs, const s_SingleTargetMatch& rhs) { return lhs.dMatchScore > rhs.dMatchScore; }
bool compareMatchResultByPosX(const s_SingleTargetMatch& lhs, const s_SingleTargetMatch& rhs) { return lhs.ptCenter.x < rhs.ptCenter.x; }

void FilterWithRotatedRect(std::vector<s_MatchParameter>* vec, int iMethod, double dMaxOverLap)
{
	int iMatchSize = (int)vec->size();
	cv::RotatedRect rect1, rect2;
	for (int i = 0; i < iMatchSize - 1; i++)
	{
		if (vec->at(i).bDelete)
			continue;
		for (int j = i + 1; j < iMatchSize; j++)
		{
			if (vec->at(j).bDelete)
				continue;
			rect1 = vec->at(i).rectR;
			rect2 = vec->at(j).rectR;
			std::vector<cv::Point2f> vecInterSec;
			int iInterSecType = rotatedRectangleIntersection(rect1, rect2, vecInterSec);
			if (iInterSecType == cv::INTERSECT_NONE)//無交集
				continue;
			else if (iInterSecType == cv::INTERSECT_FULL) //一個矩形包覆另一個
			{
				int iDeleteIndex;
				if (iMethod == CV_TM_SQDIFF)
					iDeleteIndex = (vec->at(i).dMatchScore <= vec->at(j).dMatchScore) ? j : i;
				else
					iDeleteIndex = (vec->at(i).dMatchScore >= vec->at(j).dMatchScore) ? j : i;
				vec->at(iDeleteIndex).bDelete = true;
			}
			else//交點 > 0
			{
				if (vecInterSec.size() < 3)//一個或兩個交點
					continue;
				else
				{
					int iDeleteIndex;
					//求面積與交疊比例
					SortPtWithCenter(vecInterSec);
					double dArea = contourArea(vecInterSec);
					double dRatio = dArea / rect1.size.area();
					//若大於最大交疊比例，選分數高的
					if (dRatio > dMaxOverLap)
					{
						if (iMethod == CV_TM_SQDIFF)
							iDeleteIndex = (vec->at(i).dMatchScore <= vec->at(j).dMatchScore) ? j : i;
						else
							iDeleteIndex = (vec->at(i).dMatchScore >= vec->at(j).dMatchScore) ? j : i;
						vec->at(iDeleteIndex).bDelete = true;
					}
				}
			}
		}
	}
	std::vector<s_MatchParameter>::iterator it;
	for (it = vec->begin(); it != vec->end();)
	{
		if ((*it).bDelete)
			it = vec->erase(it);
		else
			++it;
	}
}


void GetRotatedROI(cv::Mat& matSrc, cv::Size size, cv::Point2f ptLT, double dAngle, cv::Mat& matROI)
{
	double dAngle_radian = dAngle * D2R;
	cv::Point2f ptC((matSrc.cols - 1) / 2.0f, (matSrc.rows - 1) / 2.0f);
	cv::Point2f ptLT_rotate = ptRotatePt2f(ptLT, ptC, dAngle_radian);
	cv::Size sizePadding(size.width + 6, size.height + 6);


	cv::Mat rMat = getRotationMatrix2D(ptC, dAngle, 1);
	rMat.at<double>(0, 2) -= ptLT_rotate.x - 3;
	rMat.at<double>(1, 2) -= ptLT_rotate.y - 3;
	//平移旋轉矩陣(0, 2) (1, 2)的減，為旋轉後的圖形偏移，-= ptLT_rotate.x - 3 代表旋轉後的圖形往-X方向移動ptLT_rotate.x - 3
	//Debug

	//Debug
	warpAffine(matSrc, matROI, rMat, sizePadding);
}


bool SubPixEsimation(std::vector<s_MatchParameter>* vec, double* dNewX, double* dNewY, double* dNewAngle, double dAngleStep, int iMaxScoreIndex)
{
	//Az=S, (A.T)Az=(A.T)s, z = ((A.T)A).inv (A.T)s

	cv::Mat matA(27, 10, CV_64F);
	cv::Mat matZ(10, 1, CV_64F);
	cv::Mat matS(27, 1, CV_64F);

	double dX_maxScore = (*vec)[iMaxScoreIndex].pt.x;
	double dY_maxScore = (*vec)[iMaxScoreIndex].pt.y;
	double dTheata_maxScore = (*vec)[iMaxScoreIndex].dMatchAngle;
	int iRow = 0;
	/*for (int x = -1; x <= 1; x++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int theta = 0; theta <= 2; theta++)
			{*/
	for (int theta = 0; theta <= 2; theta++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				//xx yy tt xy xt yt x y t 1
				//0  1  2  3  4  5  6 7 8 9
				double dX = dX_maxScore + x;
				double dY = dY_maxScore + y;
				//double dT = (*vec)[theta].dMatchAngle + (theta - 1) * dAngleStep;
				double dT = (dTheata_maxScore + (theta - 1) * dAngleStep) * D2R;
				matA.at<double>(iRow, 0) = dX * dX;
				matA.at<double>(iRow, 1) = dY * dY;
				matA.at<double>(iRow, 2) = dT * dT;
				matA.at<double>(iRow, 3) = dX * dY;
				matA.at<double>(iRow, 4) = dX * dT;
				matA.at<double>(iRow, 5) = dY * dT;
				matA.at<double>(iRow, 6) = dX;
				matA.at<double>(iRow, 7) = dY;
				matA.at<double>(iRow, 8) = dT;
				matA.at<double>(iRow, 9) = 1.0;
				matS.at<double>(iRow, 0) = (*vec)[iMaxScoreIndex + (theta - 1)].vecResult[x + 1][y + 1];
				iRow++;
#ifdef _DEBUG
				/*string str = format ("%.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f", dValueA[0], dValueA[1], dValueA[2], dValueA[3], dValueA[4], dValueA[5], dValueA[6], dValueA[7], dValueA[8], dValueA[9]);
				fileA <<  str << endl;
				str = format ("%.6f", dValueS[iRow]);
				fileS << str << endl;*/
#endif
			}
		}
	}
	//求解Z矩陣，得到k0~k9
	//[ x* ] = [ 2k0 k3 k4 ]-1 [ -k6 ]
	//| y* | = | k3 2k1 k5 |   | -k7 |
	//[ t* ] = [ k4 k5 2k2 ]   [ -k8 ]

	//solve (matA, matS, matZ, DECOMP_SVD);
	matZ = (matA.t() * matA).inv() * matA.t() * matS;
	cv::Mat matZ_t;
	transpose(matZ, matZ_t);
	double* dZ = matZ_t.ptr<double>(0);
	cv::Mat matK1 = (cv::Mat_<double>(3, 3) <<
		(2 * dZ[0]), dZ[3], dZ[4],
		dZ[3], (2 * dZ[1]), dZ[5],
		dZ[4], dZ[5], (2 * dZ[2]));
	cv::Mat matK2 = (cv::Mat_<double>(3, 1) << -dZ[6], -dZ[7], -dZ[8]);
	cv::Mat matDelta = matK1.inv() * matK2;

	*dNewX = matDelta.at<double>(0, 0);
	*dNewY = matDelta.at<double>(1, 0);
	*dNewAngle = matDelta.at<double>(2, 0) * R2D;
	return true;
}


void FilterWithScore(std::vector<s_MatchParameter>* vec, double dScore)
{
	std::cout << "Filtering with score: " << dScore << std::endl;
	sort(vec->begin(), vec->end(), compareScoreBig2Small);
	int iSize = vec->size(), iIndexDelete = iSize + 1;
	for (int i = 0; i < iSize; i++)
	{
		if ((*vec)[i].dMatchScore < dScore)
		{
			iIndexDelete = i;
			break;
		}
	}
	if (iIndexDelete == iSize + 1)//沒有任何元素小於dScore
		return;
	vec->erase(vec->begin() + iIndexDelete, vec->end());
	return;
}

void OutputRoi(s_SingleTargetMatch sstm)
{
	// CBB
	bool m_ckOutputROI = false;
	//if (!m_ckOutputROI.GetCheck())
	if (!m_ckOutputROI)
		return;
	cv::Rect rect(sstm.ptLT, sstm.ptRB);
	for (int i = 1; i < 50; i++)
	{
		std::string strName = cv::format("C:\\Users\\Dennis\\Desktop\\testImage\\MatchFail\\workSpace\\roi%d.bmp", i);
		//if (::PathFileExists(CString(strName.c_str())))
		//	continue;
		// CBB
		//imwrite(strName, m_matSrc(rect));
		break;
	}
}













void LearnPattern(cv::Mat& m_matDst, s_TemplData& m_TemplData, int m_iMinReduceArea)
{
	m_TemplData.clear();

	int iTopLayer = GetTopLayer(&m_matDst, (int)sqrt((double)m_iMinReduceArea));
	buildPyramid(m_matDst, m_TemplData.vecPyramid, iTopLayer);
	s_TemplData* templData = &m_TemplData;
	templData->iBorderColor = mean(m_matDst).val[0] < 128 ? 255 : 0;
	int iSize = templData->vecPyramid.size();
	templData->resize(iSize);

	for (int i = 0; i < iSize; i++)
	{
		double invArea = 1. / ((double)templData->vecPyramid[i].rows * templData->vecPyramid[i].cols);
		cv::Scalar templMean, templSdv;
		double templNorm = 0, templSum2 = 0;

		meanStdDev(templData->vecPyramid[i], templMean, templSdv);
		templNorm = templSdv[0] * templSdv[0] + templSdv[1] * templSdv[1] + templSdv[2] * templSdv[2] + templSdv[3] * templSdv[3];

		if (templNorm < DBL_EPSILON)
		{
			templData->vecResultEqual1[i] = true;
		}
		templSum2 = templNorm + templMean[0] * templMean[0] + templMean[1] * templMean[1] + templMean[2] * templMean[2] + templMean[3] * templMean[3];


		templSum2 /= invArea;
		templNorm = std::sqrt(templNorm);
		templNorm /= std::sqrt(invArea); // care of accuracy here


		templData->vecInvArea[i] = invArea;
		templData->vecTemplMean[i] = templMean;
		templData->vecTemplNorm[i] = templNorm;
	}
	templData->bIsPatternLearned = true;
	std::cout << "learned pattern? " << templData->vecInvArea[templData->vecInvArea.size() - 1] << " " << std::endl;

}











bool Match(cv::Mat& m_matSrc, cv::Mat& m_matDst)
{
	s_TemplData m_TemplData;


	// CBB this may be diff
	int m_iMaxPos = 10;
	double m_dMaxOverlap = 0.25;
	double m_dScore = 0.1;
	double m_dToleranceAngle = 180;
	int m_iMinReduceArea = 256;
	int m_iMessageCount;
	double m_dTolerance1(40);
	double m_bStopLayer1(false);
	double m_dTolerance3(-110);
	double m_dTolerance4(-100);
	double m_dTolerance2(60);
	// CBB check this
	bool m_ckBitwiseNot = false;
	bool mDebugMode = false;

	std::vector<s_SingleTargetMatch> m_vecSingleTargetData;
	bool m_bToleranceRange = false;

	LearnPattern(m_matDst, m_TemplData, m_iMinReduceArea);


	if (m_matSrc.empty() || m_matDst.empty())
	{
		std::cout << "One of the input matrixes was empyty" << std::endl;
		return false;
	}
	if ((m_matDst.cols < m_matSrc.cols && m_matDst.rows > m_matSrc.rows) || (m_matDst.cols > m_matSrc.cols && m_matDst.rows < m_matSrc.rows))
	{
		std::cout << "rows + cols issue" << std::endl;
		return false;
	}
	if (m_matDst.size().area() > m_matSrc.size().area())
	{
		std::cout << "size issue" << std::endl;
		return false;
	}
	if (!m_TemplData.bIsPatternLearned)
	{
		std::cout << "already learned" << std::endl;
		return false;
	}
	//UpdateData(1);
	double d1 = clock();
	//??????? ???1 + iLayer?
	int iTopLayer = GetTopLayer(&m_matDst, (int)sqrt((double)m_iMinReduceArea));
	std::cout << "Top layer: " << iTopLayer << std::endl;
	std::cout << m_matSrc.rows << " " << m_matSrc.cols << " " << m_matSrc.type() << std::endl;
	std::cout << m_matDst.rows << " " << m_matDst.cols << " " << m_matDst.type() << std::endl;
	//?????
	std::vector<cv::Mat> vecMatSrcPyr;
	// CBB 
	//if (m_ckBitwiseNot.GetCheck())
	if (m_ckBitwiseNot)
	{
		std::cout << "ckBitWise TRUE" << std::endl;
		cv::Mat matNewSrc = 255 - m_matSrc;
		buildPyramid(matNewSrc, vecMatSrcPyr, iTopLayer);
		cv::imshow("1", matNewSrc);
		cv::moveWindow("1", 0, 0);
	}
	else
	{
		std::cout << "ckBitWise Flase" << std::endl;
		buildPyramid(m_matSrc, vecMatSrcPyr, iTopLayer);
	}


	s_TemplData* pTemplData = &m_TemplData;

	//???????????????ROI
	double dAngleStep = atan(2.0 / std::max(pTemplData->vecPyramid[iTopLayer].cols, pTemplData->vecPyramid[iTopLayer].rows)) * R2D;
	std::cout << "Danglestep" << dAngleStep << std::endl;
	std::vector<double> vecAngles;

	if (m_bToleranceRange)
	{
		if (m_dTolerance1 >= m_dTolerance2 || m_dTolerance3 >= m_dTolerance4)
		{
			std::cout << "Angles tolerance out of range, left has to be bigger or something" << std::endl;
			return false;
		}
		for (double dAngle = m_dTolerance1; dAngle < m_dTolerance2 + dAngleStep; dAngle += dAngleStep)
			vecAngles.push_back(dAngle);
		for (double dAngle = m_dTolerance3; dAngle < m_dTolerance4 + dAngleStep; dAngle += dAngleStep)
			vecAngles.push_back(dAngle);
	}
	else
	{
		if (m_dToleranceAngle < VISION_TOLERANCE)
			vecAngles.push_back(0.0);
		else
		{
			for (double dAngle = 0; dAngle < m_dToleranceAngle + dAngleStep; dAngle += dAngleStep)
				vecAngles.push_back(dAngle);
			for (double dAngle = -dAngleStep; dAngle > -m_dToleranceAngle - dAngleStep; dAngle -= dAngleStep)
				vecAngles.push_back(dAngle);
		}
	}
	std::cout << "Num dangles: " << vecAngles.size() << std::endl;

	int iTopSrcW = vecMatSrcPyr[iTopLayer].cols, iTopSrcH = vecMatSrcPyr[iTopLayer].rows;
	cv::Point2f ptCenter((iTopSrcW - 1) / 2.0f, (iTopSrcH - 1) / 2.0f);
	std::cout << "Center point: " << ptCenter.x << " " << ptCenter.y << std::endl;
	int iSize = (int)vecAngles.size();
	//std::vector<s_MatchParameter> vecMatchParameter (iSize * (m_iMaxPos + MATCH_CANDIDATE_NUM));
	std::vector<s_MatchParameter> vecMatchParameter;
	//Caculate lowest score at every layer
	std::vector<double> vecLayerScore(iTopLayer + 1, m_dScore);
	for (int iLayer = 1; iLayer <= iTopLayer; iLayer++)
		vecLayerScore[iLayer] = vecLayerScore[iLayer - 1] * 0.9;

	cv::Size sizePat = pTemplData->vecPyramid[iTopLayer].size();
	bool bCalMaxByBlock = (vecMatSrcPyr[iTopLayer].size().area() / sizePat.area() > 500) && m_iMaxPos > 10;
	std::cout << "ISIZE SIZE: " << iSize << std::endl;
	std::cout << "bCalMaxbyBlock: " << bCalMaxByBlock << std::endl;
	for (int i = 0; i < iSize; i++)
	{
		cv::Mat matRotatedSrc, matR = getRotationMatrix2D(ptCenter, vecAngles[i], 1);
		cv::Mat matResult;
		cv::Point ptMaxLoc;
		double dValue, dMaxVal;
		double dRotate = clock();
		cv::Size sizeBest = GetBestRotationSize(vecMatSrcPyr[iTopLayer].size(), pTemplData->vecPyramid[iTopLayer].size(), vecAngles[i]);

		float fTranslationX = (sizeBest.width - 1) / 2.0f - ptCenter.x;
		float fTranslationY = (sizeBest.height - 1) / 2.0f - ptCenter.y;
		matR.at<double>(0, 2) += fTranslationX;
		matR.at<double>(1, 2) += fTranslationY;
		warpAffine(vecMatSrcPyr[iTopLayer], matRotatedSrc, matR, sizeBest, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(pTemplData->iBorderColor));

		MatchTemplate(matRotatedSrc, pTemplData, matResult, iTopLayer, false);

		if (bCalMaxByBlock)
		{
			s_BlockMax blockMax(matResult, pTemplData->vecPyramid[iTopLayer].size());
			blockMax.GetMaxValueLoc(dMaxVal, ptMaxLoc);
			if (dMaxVal < vecLayerScore[iTopLayer])
				continue;
			vecMatchParameter.push_back(s_MatchParameter(cv::Point2f(ptMaxLoc.x - fTranslationX, ptMaxLoc.y - fTranslationY), dMaxVal, vecAngles[i]));
			for (int j = 0; j < m_iMaxPos + MATCH_CANDIDATE_NUM - 1; j++)
			{
				ptMaxLoc = GetNextMaxLoc(matResult, ptMaxLoc, pTemplData->vecPyramid[iTopLayer].size(), dValue, m_dMaxOverlap, blockMax);
				if (dValue < vecLayerScore[iTopLayer])
					break;
				vecMatchParameter.push_back(s_MatchParameter(cv::Point2f(ptMaxLoc.x - fTranslationX, ptMaxLoc.y - fTranslationY), dValue, vecAngles[i]));
			}
		}
		else
		{
			minMaxLoc(matResult, 0, &dMaxVal, 0, &ptMaxLoc);
			std::cout << "SIZEBEST " << i << " " << sizeBest.height << " " << sizeBest.width << " " << dMaxVal << " " << ptMaxLoc.x << std::endl;
			if (dMaxVal < vecLayerScore[iTopLayer])
				continue;
			vecMatchParameter.push_back(s_MatchParameter(cv::Point2f(ptMaxLoc.x - fTranslationX, ptMaxLoc.y - fTranslationY), dMaxVal, vecAngles[i]));
			for (int j = 0; j < m_iMaxPos + MATCH_CANDIDATE_NUM - 1; j++)
			{
				ptMaxLoc = GetNextMaxLoc(matResult, ptMaxLoc, pTemplData->vecPyramid[iTopLayer].size(), dValue, m_dMaxOverlap);
				if (dValue < vecLayerScore[iTopLayer])
					break;
				vecMatchParameter.push_back(s_MatchParameter(cv::Point2f(ptMaxLoc.x - fTranslationX, ptMaxLoc.y - fTranslationY), dValue, vecAngles[i]));
			}
		}
	}
	std::sort(vecMatchParameter.begin(), vecMatchParameter.end(), compareScoreBig2Small);


	int iMatchSize = (int)vecMatchParameter.size();
	int iDstW = pTemplData->vecPyramid[iTopLayer].cols, iDstH = pTemplData->vecPyramid[iTopLayer].rows;
	std::cout << "iMatchSize " << iMatchSize << std::endl;
	//???????
	if (mDebugMode)
	{
		int iDebugScale = 2;

		cv::Mat matShow, matResize;
		resize(vecMatSrcPyr[iTopLayer], matResize, vecMatSrcPyr[iTopLayer].size() * iDebugScale);
		cv::cvtColor(matResize, matShow, CV_GRAY2BGR);
		std::string str = cv::format("Toplayer, Candidate:%d", iMatchSize);
		std::vector<cv::Point2f> vec;
		for (int i = 0; i < iMatchSize; i++)
		{
			cv::Point2f ptLT, ptRT, ptRB, ptLB;
			double dRAngle = -vecMatchParameter[i].dMatchAngle * D2R;
			ptLT = ptRotatePt2f(vecMatchParameter[i].pt, ptCenter, dRAngle);
			ptRT = cv::Point2f(ptLT.x + iDstW * (float)cos(dRAngle), ptLT.y - iDstW * (float)sin(dRAngle));
			ptLB = cv::Point2f(ptLT.x + iDstH * (float)sin(dRAngle), ptLT.y + iDstH * (float)cos(dRAngle));
			ptRB = cv::Point2f(ptRT.x + iDstH * (float)sin(dRAngle), ptRT.y + iDstH * (float)cos(dRAngle));
			line(matShow, ptLT * iDebugScale, ptLB * iDebugScale, cv::Scalar(0, 255, 0));
			line(matShow, ptLB * iDebugScale, ptRB * iDebugScale, cv::Scalar(0, 255, 0));
			line(matShow, ptRB * iDebugScale, ptRT * iDebugScale, cv::Scalar(0, 255, 0));
			line(matShow, ptRT * iDebugScale, ptLT * iDebugScale, cv::Scalar(0, 255, 0));
			circle(matShow, ptLT * iDebugScale, 1, cv::Scalar(0, 0, 255));
			vec.push_back(ptLT * iDebugScale);
			vec.push_back(ptRT * iDebugScale);
			vec.push_back(ptLB * iDebugScale);
			vec.push_back(ptRB * iDebugScale);

			std::string strText = cv::format("%d", i);
			putText(matShow, strText, ptLT * iDebugScale, cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 0));
		}
		cvNamedWindow(str.c_str(), 0x10000000);
		cv::Rect rectShow = boundingRect(vec);
		cv::imshow(str, matShow);// (rectShow));
		//moveWindow (str, 0, 0);
	}
	//???????

	//??????
	//CBB 
	bool bSubPixelEstimation = false;
	//bool bSubPixelEstimation = m_bSubPixel.GetCheck();
	int iStopLayer = m_bStopLayer1 ? 1 : 0; //???1???????????????
	//int iSearchSize = min (m_iMaxPos + MATCH_CANDIDATE_NUM, (int)vecMatchParameter.size ());//?????????? ?????
	std::vector<s_MatchParameter> vecAllResult;
	for (int i = 0; i < (int)vecMatchParameter.size(); i++)
		//for (int i = 0; i < iSearchSize; i++)
	{
		double dRAngle = -vecMatchParameter[i].dMatchAngle * D2R;
		cv::Point2f ptLT = ptRotatePt2f(vecMatchParameter[i].pt, ptCenter, dRAngle);

		double dAngleStep = atan(2.0 / std::max(iDstW, iDstH)) * R2D;//min??max
		vecMatchParameter[i].dAngleStart = vecMatchParameter[i].dMatchAngle - dAngleStep;
		vecMatchParameter[i].dAngleEnd = vecMatchParameter[i].dMatchAngle + dAngleStep;

		if (iTopLayer <= iStopLayer)
		{
			vecMatchParameter[i].pt = cv::Point2d(ptLT * ((iTopLayer == 0) ? 1 : 2));
			vecAllResult.push_back(vecMatchParameter[i]);
		}
		else
		{
			for (int iLayer = iTopLayer - 1; iLayer >= iStopLayer; iLayer--)
			{
				//????
				dAngleStep = atan(2.0 / std::max(pTemplData->vecPyramid[iLayer].cols, pTemplData->vecPyramid[iLayer].rows)) * R2D;//min??max
				std::vector<double> vecAngles;
				//double dAngleS = vecMatchParameter[i].dAngleStart, dAngleE = vecMatchParameter[i].dAngleEnd;
				double dMatchedAngle = vecMatchParameter[i].dMatchAngle;
				if (m_bToleranceRange)
				{
					for (int i = -1; i <= 1; i++)
						vecAngles.push_back(dMatchedAngle + dAngleStep * i);
				}
				else
				{
					if (m_dToleranceAngle < VISION_TOLERANCE)
						vecAngles.push_back(0.0);
					else
						for (int i = -1; i <= 1; i++)
							vecAngles.push_back(dMatchedAngle + dAngleStep * i);
				}
				cv::Point2f ptSrcCenter((vecMatSrcPyr[iLayer].cols - 1) / 2.0f, (vecMatSrcPyr[iLayer].rows - 1) / 2.0f);
				iSize = (int)vecAngles.size();
				std::vector<s_MatchParameter> vecNewMatchParameter(iSize);
				int iMaxScoreIndex = 0;
				double dBigValue = -1;
				for (int j = 0; j < iSize; j++)
				{
					cv::Mat matResult, matRotatedSrc;
					double dMaxValue = 0;
					cv::Point ptMaxLoc;
					GetRotatedROI(vecMatSrcPyr[iLayer], pTemplData->vecPyramid[iLayer].size(), ptLT * 2, vecAngles[j], matRotatedSrc);

					MatchTemplate(matRotatedSrc, pTemplData, matResult, iLayer, true);
					//matchTemplate (matRotatedSrc, pTemplData->vecPyramid[iLayer], matResult, CV_TM_CCOEFF_NORMED);
					minMaxLoc(matResult, 0, &dMaxValue, 0, &ptMaxLoc);
					vecNewMatchParameter[j] = s_MatchParameter(ptMaxLoc, dMaxValue, vecAngles[j]);

					if (vecNewMatchParameter[j].dMatchScore > dBigValue)
					{
						iMaxScoreIndex = j;
						dBigValue = vecNewMatchParameter[j].dMatchScore;
					}
					//?????
					if (ptMaxLoc.x == 0 || ptMaxLoc.y == 0 || ptMaxLoc.x == matResult.cols - 1 || ptMaxLoc.y == matResult.rows - 1)
						vecNewMatchParameter[j].bPosOnBorder = true;
					if (!vecNewMatchParameter[j].bPosOnBorder)
					{
						for (int y = -1; y <= 1; y++)
							for (int x = -1; x <= 1; x++)
								vecNewMatchParameter[j].vecResult[x + 1][y + 1] = matResult.at<float>(ptMaxLoc + cv::Point(x, y));
					}
					//?????
				}
				if (vecNewMatchParameter[iMaxScoreIndex].dMatchScore < vecLayerScore[iLayer])
					break;
				//?????
				if (bSubPixelEstimation
					&& iLayer == 0
					&& (!vecNewMatchParameter[iMaxScoreIndex].bPosOnBorder)
					&& iMaxScoreIndex != 0
					&& iMaxScoreIndex != 2)
				{
					double dNewX = 0, dNewY = 0, dNewAngle = 0;
					SubPixEsimation(&vecNewMatchParameter, &dNewX, &dNewY, &dNewAngle, dAngleStep, iMaxScoreIndex);
					vecNewMatchParameter[iMaxScoreIndex].pt = cv::Point2d(dNewX, dNewY);
					vecNewMatchParameter[iMaxScoreIndex].dMatchAngle = dNewAngle;
				}
				//?????

				double dNewMatchAngle = vecNewMatchParameter[iMaxScoreIndex].dMatchAngle;

				//?????????(GetRotatedROI)?(0, 0)
				cv::Point2f ptPaddingLT = ptRotatePt2f(ptLT * 2, ptSrcCenter, dNewMatchAngle * D2R) - cv::Point2f(3, 3);
				cv::Point2f pt(vecNewMatchParameter[iMaxScoreIndex].pt.x + ptPaddingLT.x, vecNewMatchParameter[iMaxScoreIndex].pt.y + ptPaddingLT.y);
				//???
				pt = ptRotatePt2f(pt, ptSrcCenter, -dNewMatchAngle * D2R);

				if (iLayer == iStopLayer)
				{
					vecNewMatchParameter[iMaxScoreIndex].pt = pt * (iStopLayer == 0 ? 1 : 2);
					vecAllResult.push_back(vecNewMatchParameter[iMaxScoreIndex]);
				}
				else
				{
					//??MatchAngle ptLT
					vecMatchParameter[i].dMatchAngle = dNewMatchAngle;
					vecMatchParameter[i].dAngleStart = vecMatchParameter[i].dMatchAngle - dAngleStep / 2;
					vecMatchParameter[i].dAngleEnd = vecMatchParameter[i].dMatchAngle + dAngleStep / 2;
					ptLT = pt;
				}
			}

		}
	}
	FilterWithScore(&vecAllResult, m_dScore);

	//??????
	iDstW = pTemplData->vecPyramid[iStopLayer].cols * (iStopLayer == 0 ? 1 : 2);
	iDstH = pTemplData->vecPyramid[iStopLayer].rows * (iStopLayer == 0 ? 1 : 2);

	for (int i = 0; i < (int)vecAllResult.size(); i++)
	{
		cv::Point2f ptLT, ptRT, ptRB, ptLB;
		double dRAngle = -vecAllResult[i].dMatchAngle * D2R;
		ptLT = vecAllResult[i].pt;
		ptRT = cv::Point2f(ptLT.x + iDstW * (float)cos(dRAngle), ptLT.y - iDstW * (float)sin(dRAngle));
		ptLB = cv::Point2f(ptLT.x + iDstH * (float)sin(dRAngle), ptLT.y + iDstH * (float)cos(dRAngle));
		ptRB = cv::Point2f(ptRT.x + iDstH * (float)sin(dRAngle), ptRT.y + iDstH * (float)cos(dRAngle));
		//??????
		vecAllResult[i].rectR = cv::RotatedRect(ptLT, ptRT, ptRB);
	}
	FilterWithRotatedRect(&vecAllResult, CV_TM_CCOEFF_NORMED, m_dMaxOverlap);
	//??????

	//??????
	std::sort(vecAllResult.begin(), vecAllResult.end(), compareScoreBig2Small);

	m_vecSingleTargetData.clear();
	// CBB 
	std::vector<std::string> m_listMsg;
	//m_listMsg.DeleteAllItems();

	iMatchSize = (int)vecAllResult.size();
	std::cout << "NUM OF MATCHES: " << iMatchSize << std::endl;
	if (vecAllResult.size() == 0)
		return false;
	int iW = pTemplData->vecPyramid[0].cols, iH = pTemplData->vecPyramid[0].rows;

	for (int i = 0; i < iMatchSize; i++)
	{
		s_SingleTargetMatch sstm;
		double dRAngle = -vecAllResult[i].dMatchAngle * D2R;

		sstm.ptLT = vecAllResult[i].pt;

		sstm.ptRT = cv::Point2d(sstm.ptLT.x + iW * cos(dRAngle), sstm.ptLT.y - iW * sin(dRAngle));
		sstm.ptLB = cv::Point2d(sstm.ptLT.x + iH * sin(dRAngle), sstm.ptLT.y + iH * cos(dRAngle));
		sstm.ptRB = cv::Point2d(sstm.ptRT.x + iH * sin(dRAngle), sstm.ptRT.y + iH * cos(dRAngle));
		sstm.ptCenter = cv::Point2d((sstm.ptLT.x + sstm.ptRT.x + sstm.ptRB.x + sstm.ptLB.x) / 4, (sstm.ptLT.y + sstm.ptRT.y + sstm.ptRB.y + sstm.ptLB.y) / 4);
		sstm.dMatchedAngle = -vecAllResult[i].dMatchAngle;
		sstm.dMatchScore = vecAllResult[i].dMatchScore;
	    std::cout << "Match: " << sstm.dMatchScore << " " << sstm.ptCenter.x << " " << sstm.ptCenter.y << " " << sstm.dMatchedAngle << std::endl;


		if (sstm.dMatchedAngle < -180)
			sstm.dMatchedAngle += 360;
		if (sstm.dMatchedAngle > 180)
			sstm.dMatchedAngle -= 360;
		m_vecSingleTargetData.push_back(sstm);



		//Test Subpixel
		/*Point2d ptLT = vecAllResult[i].ptSubPixel;
		Point2d ptRT = Point2d (sstm.ptLT.x + iW * cos (dRAngle), sstm.ptLT.y - iW * sin (dRAngle));
		Point2d ptLB = Point2d (sstm.ptLT.x + iH * sin (dRAngle), sstm.ptLT.y + iH * cos (dRAngle));
		Point2d ptRB = Point2d (sstm.ptRT.x + iH * sin (dRAngle), sstm.ptRT.y + iH * cos (dRAngle));
		Point2d ptCenter = Point2d ((sstm.ptLT.x + sstm.ptRT.x + sstm.ptRB.x + sstm.ptLB.x) / 4, (sstm.ptLT.y + sstm.ptRT.y + sstm.ptRB.y + sstm.ptLB.y) / 4);
		CString strDiff;strDiff.Format (L"Diff(x, y):%.3f, %.3f", ptCenter.x - sstm.ptCenter.x, ptCenter.y - sstm.ptCenter.y);
		AfxMessageBox (strDiff);*/
		//Test Subpixel
		//??MATCH ROI
		OutputRoi(sstm);
		if (i + 1 == m_iMaxPos)
			break;
	}
	//sort (m_vecSingleTargetData.begin (), m_vecSingleTargetData.end (), compareMatchResultByPosX);
	//m_listMsg.DeleteAllItems();

	// CBB here's the results
	for (int i = 0; i < (int)m_vecSingleTargetData.size(); i++)
	{
		s_SingleTargetMatch sstm = m_vecSingleTargetData[i];
		//Msg
		std::string str("");
		//m_listMsg.InsertItem(i, str);
		//m_listMsg.SetCheck(i);
		//str.Format(L"%d", i);
		//m_listMsg.SetItemText(i, SUBITEM_INDEX, str);
		//str.Format(L"%.3f", sstm.dMatchScore);
		//m_listMsg.SetItemText(i, SUBITEM_SCORE, str);
		//str.Format(L"%.3f", sstm.dMatchedAngle);
		//m_listMsg.SetItemText(i, SUBITEM_ANGLE, str);
		//str.Format(L"%.3f", sstm.ptCenter.x);
		//m_listMsg.SetItemText(i, SUBITEM_POS_X, str);
		//str.Format(L"%.3f", sstm.ptCenter.y);
		//m_listMsg.SetItemText(i, SUBITEM_POS_Y, str);
		//Msg
	}
	//m_bShowResult = TRUE;

	// CBB this is where everything is shown.
	//RefreshSrcView();


	return (int)m_vecSingleTargetData.size();
}



int main()
{
	std::string dstImage = "C:\\Users\\Raiyan\\Desktop\\comp1\\2023_11_29_15_21_41_347_Top_Camera-comp1-target.png";
	std::string srcImage = "C:\\Users\\Raiyan\\Desktop\\panels\\2023_11_29_15_15_23_253_Top_Camera-comp1-vconcat-ref.png";
    auto src = cv::imread(srcImage, cv::IMREAD_GRAYSCALE);
    auto dst = cv::imread(dstImage, cv::IMREAD_GRAYSCALE);
	Match(src, dst);
}