-- Full table scan + join
SELECT *
FROM orders o
JOIN clients c ON o.client_id = c.id
WHERE c.lastname LIKE '%a%';

-- Filesort + no index
SELECT *
FROM products
ORDER BY price DESC;

-- Subquery abuse
SELECT *
FROM orders
WHERE client_id IN (
  SELECT id FROM clients WHERE wilaya = 16
);

-- Join explosion
SELECT *
FROM orders o
JOIN cart ca ON o.cart_id = ca.id
JOIN products p ON ca.product_id = p.id
JOIN admin a ON p.admin_id = a.id
WHERE o.orderdate BETWEEN '2022-01-01' AND '2025-01-01';
