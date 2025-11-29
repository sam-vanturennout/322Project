"""
Programmer: Cooper Braun, Sam Venturaanet
Class: CPSC 322-01, Fall 2025
Final Project
12/1/25
Description: Implements the MyPyTable class for core data wrangling tasks
needed in for our analysis. Handles loading/saving CSV files, projecting columns, tidying
missing values, and summarizing columns.
"""

import copy
import csv
from tabulate import tabulate

class PyTable:
    """
    Represents a 2D table of data with column names.
    
    Attributes:
        column_names (list of str): M column names
        data (list of list of obj): 2D data structure storing mixed type data.
            There are N rows by M columns.
    """

    def __init__(self, column_names=None, data=None):
        """
        Initializer for MyPyTable.

        Parameters:
            column_names (list of str): initial M column names (None if empty)
            data (list of list of obj): initial table data in shape NxM (None if empty)
        """
        if column_names is None:
            column_names = []
        self.column_names = copy.deepcopy(column_names)
        if data is None:
            data = []
        self.data = copy.deepcopy(data)

    def pretty_print(self):
        """
        Prints the table in a nicely formatted grid structure.
        """
        print(tabulate(self.data, headers=self.column_names))

    def get_shape(self):
        """
        Computes the dimension of the table (N x M).

        Returns:
            tuple: (N, M) where N is number of rows and M is number of columns
        """
        return (len(self.data), len(self.column_names))

    def get_column(self, col_identifier, include_missing_values=True):
        """
        Extracts a column from the table data as a list.

        Parameters:
            col_identifier (str or int): string for a column name or int
                for a column index
            include_missing_values (bool): True if missing values ("NA")
                should be included in the column, False otherwise.

        Returns:
            list of obj: 1D list of values in the column

        Raises:
            ValueError: if col_identifier is invalid
        """
        # determine column index
        if isinstance(col_identifier, int):
            col_index = col_identifier
            if col_index < 0 or (self.data and col_index >= len(self.data[0])) or col_index >= len(self.column_names):
                raise ValueError("Column index out of range.")
        elif isinstance(col_identifier, str):
            if col_identifier not in self.column_names:
                raise ValueError("Column name not found.")
            col_index = self.column_names.index(col_identifier)
        else:
            raise ValueError("Invalid column identifier type.")
        
        col = []
        for row in self.data:
            val = row[col_index]
            if include_missing_values or val != "NA":
                col.append(val)
        return col

    def convert_to_numeric(self):
        """Try to convert each value in the table to a numeric type (float).

        Notes:
            Leaves values as-is that cannot be converted to numeric.
        """
        for i, row in enumerate(self.data):
            new_row = []
            for val in row:
                if isinstance(val, str) and val.strip() == "NA":
                    new_row.append("NA")
                    continue
                try:
                    # only convert things that clearly look numeric (strings like "123", "123.4")
                    new_row.append(float(val))
                except Exception:
                    new_row.append(val)
            self.data[i] = new_row

    def drop_rows(self, row_indexes_to_drop):
        """
        Remove rows from the table data.

        Parameters:
            row_indexes_to_drop (list of int): list of row indexes to remove from the table data.
        """
        to_drop = set(row_indexes_to_drop)
        self.data = [r for index, r in enumerate(self.data) if index not in to_drop]

    def load_from_file(self, filename):
        """
        Load column names and data from a CSV file.

        Parameters:
            filename (str): relative path for the CSV file to open and load the contents of.

        Returns:
            MyPyTable: returns self so the caller can write code like
                table = MyPyTable().load_from_file(fname)

        Notes:
            Uses the csv module.
            First row of CSV file is assumed to be the header.
            Calls convert_to_numeric() after load.
        """
        with open(filename, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)
        if not rows:
            self.column_names = []
            self.data = []
            return self
        
        self.column_names = rows[0]
        self.data = []
        for r in rows[1:]:
            # keep exact cell strings (including "NA") so tests can detect them
            # numeric conversion happens after
            self.data.append(r)
        self.convert_to_numeric()
        return self

    def save_to_file(self, filename):
        """
        Save column names and data to a CSV file.

        Parameters:
            filename (str): relative path for the CSV file to save the contents to.

        Notes:
            Uses the csv module.
        """
        with open(filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(self.column_names)
            for row in self.data:
                writer.writerow(row)

    def find_duplicates(self, key_column_names):
        """
        Returns a list of indexes representing duplicate rows.
        Rows are identified uniquely based on key_column_names.

        Parameters:
            key_column_names (list of str): column names to use as row keys.

        Returns:
            list of int: list of indexes of duplicate rows found

        Notes:
            Subsequent occurrence(s) of a row are considered the duplicate(s).
            The first instance of a row is not considered a duplicate.
        """
        key_indexes = [self.column_names.index(k) for k in key_column_names]
        seen = set()
        dups = []
        for i, row in enumerate(self.data):
            key = tuple(row[idx] for idx in key_indexes)
            if key in seen:
                dups.append(i)
            else:
                seen.add(key)
        return dups

    def remove_rows_with_missing_values(self):
        """
        Remove rows from the table data that contain a missing value ("NA").
        """
        new_data = []
        for row in self.data:
            if any(val == "NA" for val in row):
                continue
            new_data.append(row)
        self.data = new_data

    def replace_missing_values_with_column_average(self, col_name):
        """
        For columns with continuous data, fill missing values in a column
        by the column's original average.

        Parameters:
            col_name (str): name of column to fill with the original average (of the column).
        """
        if col_name not in self.column_names:
            return
        c = self.column_names.index(col_name)
        numeric_vals = []
        for row in self.data:
            v = row[c]
            if v != "NA" and isinstance(v, (int, float)):
                numeric_vals.append(float(v))
        if not numeric_vals:
            return
        avg = sum(numeric_vals) / len(numeric_vals)
        for row in self.data:
            if row[c] == "NA":
                row[c] = avg

    def compute_summary_statistics(self, col_names):
        """
        Calculates summary stats for this MyPyTable and stores the stats in a new MyPyTable.
            min: minimum of the column
            max: maximum of the column
            mid: mid-value (AKA mid-range) of the column
            avg: mean of the column
            median: median of the column

        Parameters:
            col_names (list of str): names of the numeric columns to compute summary stats for.

        Returns:
            MyPyTable: stores the summary stats computed. The column names and their order
                is as follows: ["attribute", "min", "max", "mid", "avg", "median"]

        Notes:
            Missing values in the columns to compute summary stats
            should be ignored.
            Assumes col_names only contains the names of columns with numeric data.
        """
        out_cols = ["attribute", "min", "max", "mid", "avg", "median"]
        # if there are no rows at all, return an empty stats table
        if not self.data:
            return PyTable(out_cols, [])
        
        out_data = []
        for name in col_names:
            c = self.column_names.index(name)
            vals = []
            for row in self.data:
                v = row[c]
                if v != "NA" and isinstance(v, (int, float)):
                    vals.append(float(v))
            if not vals:
                out_data.append([name, "NA", "NA", "NA", "NA", "NA"])
                continue
            
            vals.sort()
            n = len(vals)
            min_v = vals[0]
            max_v = vals[-1]
            mid_v = (min_v + max_v) / 2.0
            avg_v = sum(vals) / n
            median_v = vals[n // 2] if n % 2 == 1 else (vals[n // 2 - 1] + vals[n // 2]) / 2.0
            out_data.append([name, min_v, max_v, mid_v, avg_v, median_v])
            
        return PyTable(out_cols, out_data)