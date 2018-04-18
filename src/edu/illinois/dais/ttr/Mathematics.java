package edu.illinois.dais.ttr;

import java.util.ArrayList;

/**
 * A math utility class with static methods.
 *
 * @author  Bob Carpenter
 * @since   LingPipe1.0
 */
public class Mathematics {

    // forbid instances
    private Mathematics() {
        /* no instances */
    }

    /**
     * The value of the golden ratio.  The golden ratio is defined to
     * be the value &phi; such that:
     *
     * <blockquote>
     * &phi; = (&phi; + 1) / &phi;
     * </blockquote>
     *
     * Note that this is a quadratic equation (multiply both sides by
     * &phi;) with the solution roughly <code>1.61803399</code>.
     *
     * <p>See the following for a fascinating tour of the properties
     * of the golden ratio:
     *
     * <ul>
     * <li><a href="http://mathworld.wolfram.com/GoldenRatio.html"
     *>Mathworld: Golden Ratio</a></li>
     * </ul>
     */
    public static final double GOLDEN_RATIO =  (1.0 + java.lang.Math.sqrt(5))/2.0;


    /**
     * An array of the Fibonacci sequence.  The array is defined
     * as follows:
     *
     * <blockquote><pre>
     * FIBONACCI_SEQUENCE[0] = 1
     * FIBONACCI_SEQUENCE[1] = 2
     * FIBONACCI_SEQUENCE[n+2] = FIBONACCI_SEQUENCE[n+1] + FIBONACCI_SEQUENCE[n]
     * </pre></blockquote>
     *
     * So <code>FIBONACCI_SEQUENCE[0]</code> represents the second
     * Fibonacci number in the traditional numbering.  The inital entries
     * are:
     *
     * <blockquote><code>
     * 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597,
     * 2584, ...
     * </code></blockquote>
     *
     * The length of the array is 91, and the largest value is:
     *
     * <blockquote><code>
     * FIBONACCI_SEQUENCE[90] = 7540113804746346429
     *
     * </code></blockquote>
     *
     * <P>See the following references for more information on
     * the fascinating properties of Fibonacci numbers:
     *
     * <UL>
     * <LI> <a href="http://en.wikipedia.org/wiki/Fibonacci_number">Wikipedia: Fibonacci Number</a>
     * <LI> <a href="http://mathworld.wolfram.com/FibonacciNumber.html">Mathworld: Fibonacci Number</a>
     */
    public static final long[] FIBONACCI_SEQUENCE = new long[] {
    1l,
    2l,
    3l,
    5l,
    8l,
    13l,
    21l,
    34l,
    55l,
    89l,
    144l,
    233l,
    377l,
    610l,
    987l,
    1597l,
    2584l,
    4181l,
    6765l,
    10946l,
    17711l,
    28657l,
    46368l,
    75025l,
    121393l,
    196418l,
    317811l,
    514229l,
    832040l,
    1346269l,
    2178309l,
    3524578l,
    5702887l,
    9227465l,
    14930352l,
    24157817l,
    39088169l,
    63245986l,
    102334155l,
    165580141l,
    267914296l,
    433494437l,
    701408733l,
    1134903170l,
    1836311903l,
    2971215073l,
    4807526976l,
    7778742049l,
    12586269025l,
    20365011074l,
    32951280099l,
    53316291173l,
    86267571272l,
    139583862445l,
    225851433717l,
    365435296162l,
    591286729879l,
    956722026041l,
    1548008755920l,
    2504730781961l,
    4052739537881l,
    6557470319842l,
    10610209857723l,
    17167680177565l,
    27777890035288l,
    44945570212853l,
    72723460248141l,
    117669030460994l,
    190392490709135l,
    308061521170129l,
    498454011879264l,
    806515533049393l,
    1304969544928657l,
    2111485077978050l,
    3416454622906707l,
    5527939700884757l,
    8944394323791464l,
    14472334024676221l,
    23416728348467685l,
    37889062373143906l,
    61305790721611591l,
    99194853094755497l,
    160500643816367088l,
    259695496911122585l,
    420196140727489673l,
    679891637638612258l,
    1100087778366101931l,
    1779979416004714189l,
    2880067194370816120l,
    4660046610375530309l,
    7540113804746346429l
    };

    /**
     * Returns <code>true</code> if the specified number is prime.  A
     * prime is a positive number greater than <code>1</code> with no
     * divisors other than <code>1</code> and itself, thus
     * <code>{2,3,5,7,11,13,...}</code>.
     *
     * @param num Number to test for primality.
     * @return <code>true</code> if the specified number is prime.
     */
    public static boolean isPrime(int num) {
        if (num < 2) return false;
        for (int i = 2; i <= num/2; ++i)
            if (num % i == 0) return false;
        return true;
    }

    /**
     * Returns the smallest prime number that is strictly larger than
     * the specified integer.  See {@link #isPrime(int)} for the
     * definition of primality.
     *
     * @param num Base from which to look for the next prime.
     * @return Smallest prime number strictly larget than specified
     * number.
     */
    public static int nextPrime(int num) {
        if (num < 2) return 2;
        for (int i = num + 1; ; ++i)
            if (isPrime(i)) return i;
    }

    /**
     * Converts a natural logarithm to a base 2 logarithm.
     * That is, if the input is <code><i>x</i> = ln <i>z</i></code>, then
     * the return value is <code>log<sub>2</sub> <i>z</i></code>.
     * Recall that <code>log<sub>2</sub> <i>z</i> = ln <i>z</i> / ln 2.
     *
     * @param x Natural log of value.
     * @return Log base 2 of value.
     */
    public static double naturalLogToBase2Log(double x) {
        return x / LN_2;
    }

    /**
     * Returns the log base 2 of the specivied value.
     *
     * @param x Value whose log is taken.
     * @return Log of specified value.
     */
    public static double log2(double x) {
        return naturalLogToBase2Log(java.lang.Math.log(x));
    }
    
    public static double[] abs(double[] x){
    	double[] result = new double[x.length];
    	for(int i=0; i<result.length; i++){
    		result[i] = Math.abs(x[i]);
    	}
    	return result;
    }


    /**
     * Returns the integer value of reading the specified byte as an
     * unsigned value.  The computation is carried out by subtracting
     * the minimum value, as defined by the constant {@link
     * Byte#MIN_VALUE}.
     *
     * @param b Byte to convert.
     * @return Unsigned value of specified byte.
     */
    public static int byteAsUnsigned(byte b) {
        return (b >= 0) ? (int)b : (256+(int)b);
    }

    /**
     * Returns the log (base 2) of the factorial of the specified long
     * integer.  The factorial of <code>n</code> is defined for
     * <code>n > 0</code> by:
     *
     * <blockquote><code>
     *  n!
     *  = <big><big>&Pi;</big></big><sub><sub>i < 0 <= n</sub></sub> i
     * </code></blockquote>
     *
     * Taking logs of both sides gives:
     *
     * <blockquote><code>
     *  log<sub><sub>2</sub></sub> n!
     *  = <big><big>&Sigma;</big></big><sub><sub>i < 0 <= n</sub></sub>
     *    log<sub><sub>2</sub></sub> i
     * </code></blockquote>
     *
     * By convention, 0! is taken to be 1, and hence <code>ln 0! = 0</code>.
     *
     * @param n Specified long integer.
     * @return Log of factorial of specified integer.
     * @throws IllegalArgumentException If the argument is negative.
     */
    public static double log2Factorial(long n) {
        if (n < 0) {
            String msg = "Factorials only defined for non-negative arguments."
                + " Found argument=" + n;
            throw new IllegalArgumentException(msg);
        }
        double sum = 0.0;
        for (long i = 1; i <= n; ++i)
            sum += log2(i);
        return sum;
    }

    /**
     * Returns the sum of the specified array of double values.
     *
     * @param xs Array of values to sum.
     * @return The sum of the values.
     */
    public static double sum(double[] xs) {
        double sum = 0.0;
        for (int i = 0; i < xs.length; ++i)
            sum += xs[i];
        return sum;
    }
    
    /**
     * Returns the sum of the specified array of double values.
     *
     * @param xs Array of values to sum.
     * @return The sum of the values.
     */
    public static double sum(ArrayList<Double> xs) {
        double sum = 0.0;
        for (int i = 0; i < xs.size(); ++i)
            sum += xs.get(i);
        return sum;
    }

    /**
     * Returns the minimum of the specified array of double values.
     * If the length of the array is zero, the result is {@link
     * Double#NaN}.
     *
     * @param xs Array of values.
     * @return Minimum value in array.
     */
    public static double minimum(double[] xs) {
        if (xs.length == 0) return Double.NaN;
        double min = xs[0];
        for (int i = 1; i < xs.length; ++i)
            if (xs[i] < min) min = xs[i];
        return min;
    }

    /**
     * Returns the maximum of the specified array of double values.
     * If the length of the array is zero, the result is {@link
     * Double#NaN}.
     *
     * @param xs Array of values.
     * @return Maximum value in array.
     */
    public static double maximum(double[] xs) {
        if (xs.length == 0) return Double.NaN;
        double max = xs[0];
        for (int i = 1; i < xs.length; ++i)
            if (xs[i] > max) max = xs[i];
        return max;
    }


    /**
     * Returns the log (base 2) of the binomial coefficient of the
     * specified arguments.  The binomial coefficient is equal to the
     * number of ways to choose a subset of size <code>m</code> from a
     * set of <code>n</code> objects, which is pronounced "n choose
     * m", and is given by:
     *
     * <blockquote><code>
     *   choose(n,m) = n! / ( m! * (n-m)!)
     *   <br>
     *   log<sub>2</sub> choose(n,m)
     *    = log<sub>2</sub> n - log<sub>2</sub> m
     *      - log<sub>2</sub> (n-m)
     * </code></blockquote>
     *
     * @return The log (base 2) of the binomial coefficient of the
     * specified arguments.
     */
    public static double log2BinomialCoefficient(long n, long m) {
        return log2(n) - log2(m) - log2(n-m);
    }

    /**
     * Throws an illegal argument exception if the specified double
     * value is not a finite non-negative number.  The label is
     * included in the generated exception's error message if
     * necessary.
     *
     * @param label Label of variable for exception message.
     * @param x Double value to check.
     * @throws IllegalArgumentException If the specified double value
     * is infinite, is not a number, or is negative.
     */
    public static void assertFiniteNonNegative(String label, double x) {
        if (x < 0.0 ||
            Double.isNaN(x)
            || Double.isInfinite(x)) {
            String msg = label + " must be finite and non-negative."
                + " Found " + label + "=" + x;
            throw new IllegalArgumentException(msg);
        }
    }


    /**
     * Returns the log (base 2) of the &Gamma; function.  The &Gamma; function
     * is defined by:
     *
     * <blockquote><pre>
     * &Gamma;(z) = <big><big><big><big>&#8747;</big></big></big></big><sub><sub><sub><big>0</big></sub></sub></sub><sup><sup><sup><big>&#8734;</big></sup></sup></sup> t<sup>z-1</sup> * e<sup>-t</sup> <i>d</i>t</pre></blockquote>
     *
     * <p>The &Gamma; function is the continuous generalization of the factorial
     * function, so that for real numbers <code>z &gt; 0</code>:
     *
     * <blockquote><code>&Gamma;(z+1) = z * &Gamma;(z)</code></blockquote>
     *
     * In particular, integers <code>n &gt;= 0</code>, we have:
     *
     * <blockquote><code>&Gamma;(n+1) = n!</code></blockquote>
     *
     * <p>In general, &Gamma; satisfies:
     *
     * <blockquote><pre>
     * &Gamma;(z) = &pi; / (sin(&pi; * z) * &Gamma;(1-z))
     * </pre></blockquote>
     *
     * <p>This method uses the Lanczos approximation which is accurate
     * nearly to the full power of double-precision arithmetic.  The
     * Lanczos approximation is used for inputs in the range
     * <code>[0.5,1.5]</code>, converting numbers less than 0.5 using
     * the above formulas, and reducing arguments greater than 1.5
     * using the factorial-like expansion above.
     *
     * <p>For more information on the &Gamma; function and its computation, see:
     *
     * <ul>
     * <li>Weisstein, Eric W. <a href="http://mathworld.wolfram.com/GammaFunction.html">Gamma Function</a>.
     *   From MathWorld--A Wolfram Web Resource.
     * </li>
     * <li>
     * Weisstein, Eric W. <a href="http://mathworld.wolfram.com/LanczosApproximation.html">Lanczos Approximation</a>. From MathWorld--A Wolfram Web Resource.

     * <li>Wikipedia. <a href="http://en.wikipedia.org/wiki/Gamma_function">Gamma Function</a>.</li>
     * <li>Wikipedia. <a href="http://en.wikipedia.org/wiki/Lanczos_approximation">Lanczos Approximation</a>.</li>
     * </ul>
     *
     * @param z The argument to the gamma function.
     * @return The value of <code>&Gamma;(z)</code>.
     */
    public static double log2Gamma(double z) {
        if (z < 0.5) {
            return Mathematics.log2(java.lang.Math.PI)
                - Mathematics.log2(java.lang.Math.sin(java.lang.Math.PI * z))
                - log2Gamma(1.0 - z);
        }
        double result = 0.0;
        while (z > 1.5) {
            result += Mathematics.log2(z - 1);
            z -= 1.0;
        }
        return result + Mathematics.log2(lanczosGamma(z));
    }

    static double[] LANCZOS_COEFFS = new double[] {
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7
    };

    static double SQRT_2_PI = java.lang.Math.sqrt(2.0 * java.lang.Math.PI);

    // assumes input in [0.5,1.5] inclusive
    static double lanczosGamma(double z) {
        double zMinus1 = z - 1;
        double x = LANCZOS_COEFFS[0];
        for (int i = 1; i < LANCZOS_COEFFS.length - 2; ++i)
            x += LANCZOS_COEFFS[i] / (zMinus1 + i);
        double t = zMinus1 + (LANCZOS_COEFFS.length - 2) + 0.5;
        return SQRT_2_PI
            * java.lang.Math.pow(t, zMinus1 + 0.5)
            * java.lang.Math.exp(-t) * x;
    }

    /**
     * Returns the value of the digamma function for the specified
     * value.  The returned values are accurate to at least 13
     * decimal places.
     * 
     * <p>The digamma function is the derivative of the log of the
     * gamma function; see the method documentation for {@link
     * #log2Gamma(double)} for more information on the gamma function
     * itself.
     *
     * <blockquote><pre>
     * &Psi;(z)
     * = <i>d</i> log &Gamma;(z) / <i>d</i>z 
     * = &Gamma;'(z) / &Gamma;(z)
     * </pre></blockquote>
     *
     * <p>The numerical approximation is derived from:
     *
     * <ul>
     * <li>Richard J. Mathar. 2005. 
     * <a href="http://arxiv.org/abs/math.CA/0403344">Chebyshev Series Expansion of Inverse Polynomials</a>.
     * <li>
     * <li>Richard J. Mathar. 2005.
     * <a href="http://www.strw.leidenuniv.nl/~mathar/progs/digamma.c">digamma.c</a>.
     * (C Program implementing algorithm.)
     * </li>
     * </ul>
     *
     * <i>Implementation Note:</i> The recursive calls in the C
     * implementation have been transformed into loops and
     * accumulators, and the recursion for values greater than three
     * replaced with a simpler reduction.  The number of loops
     * required before the fixed length expansion is approximately
     * integer value of the absolute value of the input.  Each loop
     * requires a floating point division, two additions and a local
     * variable assignment.  The fixed portion of the algorithm is
     * roughly 30 steps requiring four multiplications, three
     * additions, one static final array lookup, and four assignments per
     * loop iteration.
     *
     * @param x Value at which to evaluate the digamma function.
     * @return The value of the digamma function at the specified
     * value.
     */
    public static double digamma(double x)
    {
	if (x <= 0.0 && (x == (double)((long) x)))
	    return Double.NaN;

	double accum = 0.0;
	if (x < 0.0) {
	    accum += java.lang.Math.PI
		/ java.lang.Math.tan(java.lang.Math.PI * (1.0 - x));
	    x = 1.0 - x;
	}

	if (x < 1.0 ) {
	    while (x < 1.0)
		accum -= 1.0 / x++;
	}

	if (x == 1.0)
	    return accum - NEGATIVE_DIGAMMA_1;

	if (x == 2.0)
	    return accum + 1.0 - NEGATIVE_DIGAMMA_1;

	if (x == 3.0)
	    return accum + 1.5 - NEGATIVE_DIGAMMA_1;

	// simpler recursion than Mahar to reduce recursion
	if (x > 3.0) {
	    while (x > 3.0)
		accum += 1.0 / --x;
	    return accum + digamma(x);
	}

	x -= 2.0;
	double tNMinus1 = 1.0;	
	double tN = x;
	double digamma = DIGAMMA_COEFFS[0] + DIGAMMA_COEFFS[1] * tN;
	for (int n = 2; n < DIGAMMA_COEFFS.length; n++) {
	    double tN1 = 2.0 * x * tN - tNMinus1;	
	    digamma += DIGAMMA_COEFFS[n] * tN1;
	    tNMinus1 = tN;
	    tN = tN1;
	}
	return accum + digamma;
    }


    /**
     * The &gamma; constant for computing the digamma function.
     * 
     * <p>The value is defined as the negative of the digamma funtion
     * evaluated at 1:
     *
     * <blockquote><pre>
     * &gamma; = - &Psi;(1)
     * 
     */
    static double NEGATIVE_DIGAMMA_1 = 0.5772156649015328606065120900824024;


    /**
     * Returns the value of the digamma function for the specified
     * value and number of iterations of the series approximation.
     * 
     * <p>The digamma function is the derivative of the log of the
     * gamma function (defined in method {@link #log2Gamma(double)}:
     *
     * <blockquote><pre>
     * &Psi;(z)
     * = <i>d</i> log &Gamma;(z) / <i>d</i>z 
     * = &Gamma;'(z) / &Gamma;(z)
     * </pre></blockquote>
     *
     * <p>The numerical approximation is derived from  formula 6.3.16 of:
     *
     * <ul>
     * <li>Abramowitz and Stegun.
     * <a href="http://www.math.sfu.ca/~cbm/aands/page_259.htm">Handbook of Mathematical Functions</i>.
     *
     * <li><a href="http://www.strw.leidenuniv.nl/~mathar/progs/digamma.c"></a></li>
     * </ul>
     *
     * The algorithm requires on the order of 50 terms for accuracy +/- 0.1,
     * 200 terms for accuracy +/- 0.01, 2000 terms for accuracy +/- 0.001,
     * 20,000 terms for accuracy +/- 0.0001, and 200,000 terms for accuracy
     * +/0 0.00001.
     *
     * @param x Value at which to evaluate the digamma function.
     * @param numTerms Number of terms in the series expansion to use.
     * @return The value of the digamma function at the specified
     * value.
     */
    public static double digamma2(double x, int numTerms) {

	// not defined for 0 or negative integers??
	if (x <= 0.0 && (x == (double)((long) x)))
	    return Double.NaN;

	double digamma = -NEGATIVE_DIGAMMA_1;

	// reflection if less than 0
	if (x < 0.0) {
	    digamma += java.lang.Math.PI
		/ java.lang.Math.tan(java.lang.Math.PI * (1.0 - x));

	    x = 1.0 - x;
	}

	if (x < 1.0) {
	    digamma -= 1.0 / x;
	    x += 1.0;
	}

	if (x == 1.0)
	    return digamma;
	
	if (x == 2.0)
	    return digamma + 1.0;
	
	if (x == 3.0)
	    return digamma + 1.5;
	
	while (x > 3.0) {
	    digamma += 1.0 / (x - 1.0);
	    x -= 1.0;
	}

	x -= 1.0;
	    
	for (int n = 1; n <= numTerms; ++n)
	    digamma += x / (n * (n + x));

	return digamma;
    }


    /**
     * The natural logarithm of 2.
     */
    public static final double LN_2 = java.lang.Math.log(2.0);

    /**
     * The log base 2 of the constant <i>e</i>, which is the base of
     * the natural logarithm.  The constant <i>e</i> is determined by
     * the java constant {@link java.lang.Math#E}.
     */
    public static final double LOG2_E = log2(java.lang.Math.E);


    private static final double DIGAMMA_COEFFS[] 
	= { 
	.30459198558715155634315638246624251,
	.72037977439182833573548891941219706, 
	-.12454959243861367729528855995001087,
	.27769457331927827002810119567456810e-1, 
	-.67762371439822456447373550186163070e-2,
	.17238755142247705209823876688592170e-2, 
	-.44817699064252933515310345718960928e-3,
	.11793660000155572716272710617753373e-3, 
	-.31253894280980134452125172274246963e-4,
	.83173997012173283398932708991137488e-5, 
	-.22191427643780045431149221890172210e-5,
	.59302266729329346291029599913617915e-6, 
	-.15863051191470655433559920279603632e-6,
	.42459203983193603241777510648681429e-7, 
	-.11369129616951114238848106591780146e-7,
	.304502217295931698401459168423403510e-8, 
	-.81568455080753152802915013641723686e-9,
	.21852324749975455125936715817306383e-9, 
	-.58546491441689515680751900276454407e-10,
	.15686348450871204869813586459513648e-10, 
	-.42029496273143231373796179302482033e-11,
	.11261435719264907097227520956710754e-11, 
	-.30174353636860279765375177200637590e-12,
	.80850955256389526647406571868193768e-13, 
	-.21663779809421233144009565199997351e-13,
	.58047634271339391495076374966835526e-14, 
	-.15553767189204733561108869588173845e-14,
	.41676108598040807753707828039353330e-15, 
	-.11167065064221317094734023242188463e-15 };



}

