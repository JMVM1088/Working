import { StockData, PivotPoint, PatternResult, AlgorithmType } from '../types';

// Helper for Linear Regression
const linearRegression = (points: {x: number, y: number}[]) => {
    const n = points.length;
    if (n < 2) return null;

    let sumX = 0, sumY = 0, sumXY = 0, sumXX = 0;
    for (const p of points) {
        sumX += p.x;
        sumY += p.y;
        sumXY += p.x * p.y;
        sumXX += p.x * p.x;
    }

    const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
    const intercept = (sumY - slope * sumX) / n;
    
    return { slope, intercept };
};

// Helper to calculate Rolling SMA and Standard Deviation
const calculateRollingStats = (values: number[], period: number) => {
    const sma = new Array(values.length).fill(0);
    const stdDev = new Array(values.length).fill(0);

    for (let i = 0; i < values.length; i++) {
        if (i < period - 1) {
            // Not enough data for full period, use expanding window or just 0
            // Using expanding window for simplicity at start
            const slice = values.slice(0, i + 1);
            const mean = slice.reduce((a, b) => a + b, 0) / slice.length;
            sma[i] = mean;
            
            const variance = slice.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / slice.length;
            stdDev[i] = Math.sqrt(variance);
        } else {
            const slice = values.slice(i - period + 1, i + 1);
            const mean = slice.reduce((a, b) => a + b, 0) / period;
            sma[i] = mean;

            const variance = slice.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / period;
            stdDev[i] = Math.sqrt(variance);
        }
    }
    return { sma, stdDev };
};

export const analyzePattern = (
    data: StockData[], 
    anchorDate: string, 
    lookbackDays: number = 90, 
    algoType: AlgorithmType,
    pivotWindow: number,
    stdDevMultiplier: number = 1.0,
    smaPeriod: number = 20
): PatternResult => {
    
    // 1. Slice Data up to Anchor Date
    const anchorIdx = data.findIndex(d => d.Date === anchorDate);
    if (anchorIdx === -1) return { found: false, pivotsHigh: [], pivotsLow: [], maturity: 0, widthStart: 0, widthEnd: 0 };

    const startIndex = Math.max(0, anchorIdx - lookbackDays + 1);
    const windowData = data.slice(startIndex, anchorIdx + 1);
    
    // --- STATISTICS CALCULATION ---
    // We calculate stats on Close price usually, or High/Low depending on strictness.
    // Standard Bollinger Bands use Close.
    const closes = windowData.map(d => d.Close);
    const { sma, stdDev } = calculateRollingStats(closes, smaPeriod);

    // --- PIVOT DETECTION ---
    const pivotsHigh: PivotPoint[] = [];
    const pivotsLow: PivotPoint[] = [];

    for (let i = pivotWindow; i < windowData.length - pivotWindow; i++) {
        const currentHigh = windowData[i].High;
        const currentLow = windowData[i].Low;
        const currentStatsIndex = i; // Map 1:1 since stats are calc on windowData
        
        // Check High Structure (Local Max)
        let isHigh = true;
        for (let j = 1; j <= pivotWindow; j++) {
            if (windowData[i - j].High > currentHigh || windowData[i + j].High > currentHigh) {
                isHigh = false;
                break;
            }
        }
        
        if (isHigh && algoType === AlgorithmType.ENHANCED) {
            // Statistical Check: Price must be significantly high
            // Condition: High > SMA + (Multiplier * StdDev)
            const upperThreshold = sma[currentStatsIndex] + (stdDevMultiplier * stdDev[currentStatsIndex]);
            if (currentHigh < upperThreshold) {
                isHigh = false;
            }
        }

        if (isHigh) {
            pivotsHigh.push({ 
                index: i, 
                date: windowData[i].Date, 
                value: currentHigh, 
                type: 'high' 
            });
        }

        // Check Low Structure (Local Min)
        let isLow = true;
        for (let j = 1; j <= pivotWindow; j++) {
            if (windowData[i - j].Low < currentLow || windowData[i + j].Low < currentLow) {
                isLow = false;
                break;
            }
        }

        if (isLow && algoType === AlgorithmType.ENHANCED) {
            // Statistical Check: Price must be significantly low
            // Condition: Low < SMA - (Multiplier * StdDev)
            const lowerThreshold = sma[currentStatsIndex] - (stdDevMultiplier * stdDev[currentStatsIndex]);
            if (currentLow > lowerThreshold) {
                isLow = false;
            }
        }

        if (isLow) {
            pivotsLow.push({ 
                index: i, 
                date: windowData[i].Date, 
                value: currentLow, 
                type: 'low' 
            });
        }
    }

    if (pivotsHigh.length < 2 || pivotsLow.length < 2) {
        return { found: false, pivotsHigh, pivotsLow, maturity: 0, widthStart: 0, widthEnd: 0 };
    }

    // --- LINE CONSTRUCTION ---
    let mRes = 0, cRes = 0, mSup = 0, cSup = 0;

    if (algoType === AlgorithmType.LEGACY) {
        // ORIGINAL LOGIC: Connect Global Max/Min to Last Pivot
        
        // Highs
        let maxVal = -Infinity;
        let maxIdx = -1;
        
        pivotsHigh.forEach(p => {
            if (p.value > maxVal) {
                maxVal = p.value;
                maxIdx = p.index;
            }
        });

        const lastPivotHigh = pivotsHigh[pivotsHigh.length - 1];
        let p2High = lastPivotHigh;
        if (lastPivotHigh.index === maxIdx && pivotsHigh.length > 1) {
            p2High = pivotsHigh[pivotsHigh.length - 2];
        }

        if (maxIdx !== -1 && p2High.index !== maxIdx) {
            mRes = (p2High.value - maxVal) / (p2High.index - maxIdx);
            cRes = maxVal - mRes * maxIdx;
        }

        // Lows
        let minVal = Infinity;
        let minIdx = -1;
        pivotsLow.forEach(p => {
            if (p.value < minVal) {
                minVal = p.value;
                minIdx = p.index;
            }
        });

        const lastPivotLow = pivotsLow[pivotsLow.length - 1];
        let p2Low = lastPivotLow;
        if (lastPivotLow.index === minIdx && pivotsLow.length > 1) {
             p2Low = pivotsLow[pivotsLow.length - 2];
        }

        if (minIdx !== -1 && p2Low.index !== minIdx) {
            mSup = (p2Low.value - minVal) / (p2Low.index - minIdx);
            cSup = minVal - mSup * minIdx;
        }

    } else {
        // ENHANCED LOGIC: Best Fit Line (Regression on Pivot Points)
        const highPoints = pivotsHigh.map(p => ({ x: p.index, y: p.value }));
        const lowPoints = pivotsLow.map(p => ({ x: p.index, y: p.value }));

        const resLine = linearRegression(highPoints);
        const supLine = linearRegression(lowPoints);

        if (resLine) { mRes = resLine.slope; cRes = resLine.intercept; }
        if (supLine) { mSup = supLine.slope; cSup = supLine.intercept; }
    }

    // --- CONVERGENCE CHECK ---
    const currentDayIdx = windowData.length - 1; // Relative index of anchor
    const startX = 0;
    
    const yResStart = mRes * startX + cRes;
    const ySupStart = mSup * startX + cSup;
    const yResEnd = mRes * currentDayIdx + cRes;
    const ySupEnd = mSup * currentDayIdx + cSup;

    const widthStart = yResStart - ySupStart;
    const widthEnd = yResEnd - ySupEnd;

    // Logic: Must be narrowing
    let isConverging = widthEnd < widthStart && widthEnd > 0;

    // Apex Calculation
    let apexX = 0;
    if (Math.abs(mRes - mSup) > 1e-9) {
        apexX = (cSup - cRes) / (mRes - mSup);
    }
    
    const maturity = currentDayIdx / apexX;

    if (algoType === AlgorithmType.ENHANCED) {
        // Lenient convergence check for flags/pennants
        isConverging = widthEnd < widthStart;
    }

    return {
        found: isConverging,
        resistanceLine: { slope: mRes, intercept: cRes, startIdx: startIndex, endIdx: anchorIdx },
        supportLine: { slope: mSup, intercept: cSup, startIdx: startIndex, endIdx: anchorIdx },
        pivotsHigh,
        pivotsLow,
        maturity,
        widthStart,
        widthEnd
    };
};