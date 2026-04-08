#define radius2 (1.0)   // (interaction range)^2
#define my_min(A, B) ((A)>(B) ? (B):(A))
#define nThreadMax 128

struct Vec2 {
    double x, y;

    __host__ __device__ Vec2() : x(0.0), y(0.0) {}
    __host__ __device__ Vec2(double x_, double y_) : x(x_), y(y_) {}

    __host__ __device__ Vec2 operator+(const Vec2& other) const { return {x+other.x, y+other.y}; }
    __host__ __device__ Vec2 operator-(const Vec2& other) const { return {x-other.x, y-other.y}; }
    __host__ __device__ Vec2 operator*(double s) const { return {x*s, y*s}; }
    __host__ __device__ Vec2& operator+=(const Vec2& o){ x+=o.x; y+=o.y; return *this; }
    __host__ __device__ double norm() const { return sqrt(x*x + y*y); }
    __host__ __device__ double sq() const { return x*x + y*y; }
    __host__ __device__ Vec2 normalized() const { double n = norm(); return {x/n, y/n}; }
};

struct Vec2Plus {
    __host__ __device__ Vec2 operator()(const Vec2& a, const Vec2& b) const {
        return Vec2(a.x + b.x, a.y + b.y);
    }
};

struct AngleToVec2 {
    __host__ __device__ Vec2 operator()(double a) const {
        return Vec2(::cos(a), ::sin(a)); 
    }
};


__host__ __device__
Vec2 operator*(const double& a, const Vec2& b) {
    return b * a;
}

__host__ __device__
Vec2 apply_periodic_boundary(const Vec2& pos, int Lx, int Ly) {
    double new_x = pos.x;
    double new_y = pos.y;

    if (new_x < 0) new_x += (double)Lx;
    else if (new_x >= Lx) new_x -= (double)Lx;
    if (new_y < 0) new_y += (double)Ly;
    else if (new_y >= Ly) new_y -= (double)Ly;

    return {new_x, new_y};
}

__device__ 
Vec2 compute_distance(const Vec2& pos1, const Vec2& pos2, int Lx, int Ly) {
    Vec2 dist = pos1 - pos2;
    dist.x -= Lx * round(dist.x / Lx);
    dist.y -= Ly * round(dist.y / Ly);
    return dist;
}

__device__ 
double dot(const Vec2& vec1, const Vec2& vec2){
    return vec1.x*vec2.x+vec1.y*vec2.y;
}
