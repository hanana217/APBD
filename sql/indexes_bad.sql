-- Missing indexes on foreign keys
DROP INDEX IF EXISTS idx_clients_wilaya ON clients;
DROP INDEX IF EXISTS idx_orders_client ON orders;

-- Useless indexes
CREATE INDEX idx_products_price ON products(price);
CREATE INDEX idx_orders_status ON orders(status);

-- Composite index done WRONG (order inverted)
CREATE INDEX idx_orders_price_date ON orders(price, orderdate);
