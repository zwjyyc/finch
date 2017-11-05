![postgresql-logo](https://www.postgresql.org/media/img/about/press/slonik_with_black_text_and_tagline.gif)

* SELECT
	* DISTINCT
		* In a table, a column may contain many duplicated values; and sometimes you only want to list the different (distinct) values. The `DISTINCT` keyword can be used to return only distinct (different) values.
		
				SELECT DISTINCT column_1, column_2 FROM table_name;
		* Example
		
				SELECT DISTINCT rating FROM film;
	* WHERE
		* To query just particular rows from a table, you need `WHERE` clause in the `SELECT` statement.
		
				SELECT column_1, column_2, ... column_n
				FROM table_name
				WHERE conditions;
		* Example
		
				SELECT customer_id,amount,payment_date FROM payment
				WHERE amount<=1 or amount>=8;

	* COUNT
		* The `COUNT(*)` function returns the number of rows returned by a SELECT clause; When you apply the `COUNT(*)` to the entire table, Postgresql scans table sequentially.
		
				SELECT COUNT(*) FROM table;
		* You can also specify a specific column count for readability
		
				SELECT COUNT(column) FROM table;
		* We can use `COUNT` with `DISTINCT`, for example:
		
				SELECT COUNT(DISTINCT column) FROM table;
		* Example
		
				SELECT COUNT(DISTINCT amount) FROM payment WHERE amount<=1 or amount>=8;

	* LIMIT
		* `LIMIT` allows you to limit the number of rows you get back after a query, useful when wanting to get all columns but not all rows
		* Example
		
				SELECT * FROM customer LIMIT 5;

	* Order By
		* When you query data from a table, PostgreSQL returns the rows in the order that they were inserted into the table. In order to sort the result set, use the `ORDER BY` clause in the `SELECT` statement. (ascending by default)
		
				SELECT column_1, column_2
				FROM table_name
				ORDER BY column 1 ASC / DESC;
		* Example
		
				SELECT first_name, last_name FROM customer ORDER BY first_name ASC;

	* Between
		* We use the `BETWEEN` operator to match a value against a range of values. For example:
		
				low <= value <= high
		* If you want to check if a value is out of a range, we use the NOT BETWEEN operator:
		
				value < low OR value > high
		* Example
		
				SELECT customer_id,amount FROM payment
				WHERE amount BETWEEN 9 AND 9;
		
				SELECT customer_id,amount FROM payment
				WHERE amount NOT BETWEEN 9 AND 9;
				
				SELECT amount,payment_date FROM payment
				WHERE payment_date NOT BETWEEN '2007-02-07' AND '2007-02-15';

	* IN
		* You use the `IN` operator with the `WHERE` clause to check if a value matches any value in a list of values. The syntax is:
		
				value IN (value1,value2,...)
		
				value IN (SELECT value FROM tbl_name)
		* You can use `NOT IN`
		
		* Example
		
				SELECT customer_id,rental_id,return_date
				FROM rental
				WHERE customer_id IN (1,2)
				ORDER BY return_date DESC;
		
				SELECT customer_id,rental_id,return_date
				FROM rental
				WHERE customer_id NOT IN (1,2)
				ORDER BY return_date DESC;
		
			It is actually equivalent to
		
				SELECT customer_id,rental_id,return_date
				FROM rental
				WHERE customer_id != 1
				OR customer_id != 2
				ORDER BY return_date DESC;
