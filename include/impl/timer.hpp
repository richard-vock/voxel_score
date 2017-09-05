#ifndef _VOXEL_SCORE_TIMER_HPP_
#define _VOXEL_SCORE_TIMER_HPP_

#include <memory>
#include <chrono>

namespace voxel_score {

class timer {
    public:
        typedef std::unique_ptr<timer>       uptr_t;
        typedef std::shared_ptr<timer>       sptr_t;
        typedef std::weak_ptr<timer>         wptr_t;
        typedef std::shared_ptr<const timer> const_sptr_t;
        typedef std::weak_ptr<const timer>   const_wptr_t;
        typedef std::chrono::high_resolution_clock clock;

    public:
        virtual ~timer() {}

        static sptr_t start() {
            sptr_t self(new timer());
            self->reset();
            return self;
        }

        template <typename DurationType>
        uint64_t
        stop() const {
            std::chrono::time_point<clock> end = clock::now();
            return std::chrono::duration_cast<DurationType>(end - begin_).count();
        }

        void
        reset() {
            begin_ = clock::now();
        }

    protected:
        timer() {}

    protected:
        std::chrono::time_point<clock> begin_;
};


} // voxel_score

#endif /* _VOXEL_SCORE_TIMER_HPP_ */
