# -*- coding: utf-8 -*-
"""
Created on Aug 12 2021

@author: rozita.teymourzadeh
"""
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

if __name__ == "__main__":
    obj = Solution()

    result = obj.lengthOfLongestSubstring("pwwkew")
    print(result)