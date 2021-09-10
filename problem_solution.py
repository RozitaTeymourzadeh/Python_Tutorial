# -*- coding: utf-8 -*-
"""
Created on Aug 12 2021

@author: rozita.teymourzadeh
"""
import math


class Solution:
    """
    3. Longest Substring Without Repeating Characters
    Input: s = "abcabcbb"
    Output: 3
    Explanation: The answer is "abc", with the length of 3.

    """
    def lengthOfLongestSubstring(self, s: str) -> int:
        def check(start, end):
            chars = [0] * 128
            for i in range(start, end + 1):
                c = s[i]
                chars[ord(c)] += 1
                if chars[ord(c)] > 1:
                    return False
            return True

        n = len(s)

        res = 0
        for i in range(n):
            for j in range(i, n):
                if check(i, j):
                    res = max(res, j - i + 1)
        return res

        """
        4. Given two sorted arrays nums1 and nums2 of size m and n respectively, 
        return the median of the two sorted arrays.
        The overall run time complexity should be O(log (m+n)).
        Input: nums1 = [1,3], nums2 = [2]
        Output: 2.00000
        Explanation: merged array = [1,2,3] and median is 2.

        """

        def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
            x = sorted(nums1 + nums2)
            length = len(x)

            if length == 0:
                median = 0
            elif length == 1:
                median = x[0]
            else:
                median_index = abs(length // 2)
                median = x[median_index]
                if length > 1 and length % 2 == 0:
                    median += x[median_index - 1]
                    median /= float(2)
            return median

        """
        5. Given a string s, return the longest palindromic substring in s.
        Input: s = "babad"
        Output: "bab"
        Note: "aba" is also a valid answer.

        """

        def longestPalindrome(self, s: str) -> str:
            self.maxlen = 0
            self.start = 0

            for i in range(len(s)):
                self.expandFromCenter(s, i, i)
                self.expandFromCenter(s, i, i + 1)
            return s[self.start:self.start + self.maxlen]

        def expandFromCenter(self, s, l, r):
            while l > -1 and r < len(s) and s[l] == s[r]:
                l -= 1
                r += 1

            if self.maxlen < r - l - 1:
                self.maxlen = r - l - 1
                self.start = l + 1

        """
        6. ZigZag Conversion
        The string "PAYPALISHIRING" is written in a zigzag pattern on a given number of rows like this: 
        (you may want to display this pattern in a fixed font for better legibility)
        And then read line by line: "PAHNAPLSIIGYIR"
        P   A   H   N
        A P L S I I G
        Y   I   R

        Input: s = "PAYPALISHIRING", numRows = 3
        Output: "PAHNAPLSIIGYIR"
        
        Input: s = "PAYPALISHIRING", numRows = 4
        Output: "PINALSIGYAHRPI"
        Explanation:
        P     I    N
        A   L S  I G
        Y A   H R
        P     I
        """

        def convert(self, s: str, numRows: int) -> str:
            if numRows == 1: return s
            res = ""
            for r in range(numRows):
                increment = (numRows - 1) * 2
                for i in range(r,len(s), increment):
                    res += s[i]
                    if (r > 0 and r < numRows -1 and i + increment - 2 * r < len(s)):
                        res += s[i + increment - 2 * r]
            return res



        """
        7. Reverse Integer
        Given a signed 32-bit integer x, return x with its digits reversed. 
        If reversing x causes the value to go outside the signed 32-bit integer 
        range [-231, 231 - 1], then return 0.
        Input: x = 123
        Output: 321
        Input: x = 0
        Output: 0    
        """

        def reverse(self, x: int) -> int:
            MIN = -2**31
            MAX = (2**31) - 1
            res = 0
            while x:
                digit = int(math.fmod(x, 10))
                x = int(x/10)
                if (res > MAX // 10 or (res == MAX //10 and digit >= MAX % 10)):
                    return 0
                if (res < MIN //10 or (res == MIN //10 and digit <= MIN % 10)):
                    return 0
                res = res * 10 + digit
            return res

        """
        8. String to Integer (atoi)
        Implement the myAtoi(string s) function, which converts a string to a 32-bit signed integer 

        Input: s = "42"
        Output: 42
        
        Condition:
        - whitespace
        - +/- symbol
        - numbers
        - between MAX_INT and MIN_INT constraints
        - random characters 
        """

        def myAtoi(self, s: str) -> int:

            result = 0
            i = 0
            negative = 1
            MAX_INT = (2 ** 31) - 1
            MIN_INT = -2 ** 31

            # check corner cases
            # Whitespace
            while i < len(s) and s[i] == ' ':
                i += 1

            # Sign
            if i < len(s) and s[i] == '-':
                i += 1
                negative = -1
            elif i <len(s) and s[i] == '+':
                i += 1
                negative = 1
            # number
            checker = set('0123456789')
            while i < len(s) and s[i] in checker:
                # check overflow
                if result > MAX_INT/10 or (result == MAX_INT/10 and int(s[i]) > 7):
                    return MAX_INT if negative == 1 else MIN_INT
                result = result * 10 + int(s[i])
                i += 1

            result = result * negative
            # check for max and Min
            if result < 0:
                return max(result, MIN_INT)
            return min(result,MAX_INT)


        
if __name__ == "__main__":
    obj = Solution()

    result = obj.lengthOfLongestSubstring("pwwkew")
    print(result)